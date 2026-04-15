"""
ChickenOrEgg - Generic Image Classifier (Ollama Edition)
=====================================================
Classify any folder of images into two categories using a local
Vision-Language Model (VLM) running via Ollama.

Supports: jpg, jpeg, png, webp, bmp, tiff, gif
Default model: qwen2.5vl:7b  (change with --model)

Requirements:
    pip install ollama pillow
    ollama pull qwen2.5vl:7b       # or any other vision model

Usage Examples:
    # Basic — classify pizzas vs not-pizzas
    python template_classifier.py \
        --source-dir ./my_images \
        --output-dir ./output \
        --target-class "Pizza" \
        --other-class "Not Pizza"

    # Specify model and file pattern
    python template_classifier.py \
        --source-dir ./my_images \
        --output-dir ./output \
        --target-class "Cat" \
        --other-class "Dog" \
        --model llava:13b \
        --file-pattern "*.png"

    # With few-shot example images for better accuracy
    python template_classifier.py \
        --source-dir ./my_images \
        --output-dir ./output \
        --target-class "Receipt" \
        --other-class "Not Receipt" \
        --example-class-a ./examples/receipt1.jpg ./examples/receipt2.jpg \
        --example-class-b ./examples/other1.jpg

Output:
    <output-dir>/
    ├── <target_class>/          ← matching images copied here
    ├── <other_class>/           ← non-matching images copied here
    ├── classification_results.json
    └── classification_report.txt
"""

import argparse
import base64
import io
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from ollama import chat

# ---------------------------------------------------------------------------
# Supported image extensions (case-insensitive)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_images(directory: Path, file_pattern=None):
    """
    Return all image files inside *directory*.

    If *file_pattern* is given (e.g. "*.jpg") only that glob pattern is used;
    otherwise every file with a recognised image extension is returned.
    """
    if file_pattern:
        files = sorted(directory.glob(file_pattern))
        return [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def encode_image(image_path, max_size=1024):
    """Resize (if needed) and return a base64-encoded JPEG string."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class GenericVLMClassifier:
    """
    Two-class image classifier powered by any Ollama vision model.

    Parameters
    ----------
    source_dir : str or Path
        Folder containing images to classify.
    output_dir : str or Path
        Folder where sorted images and reports will be written.
    target_class_name : str
        Human-readable label for the *positive* class  (e.g. "Pizza").
    other_class_name : str
        Human-readable label for the *negative* class  (e.g. "Not Pizza").
    model : str
        Ollama model tag to use for inference  (e.g. "qwen2.5vl:7b").
    example_class_a_paths : list of str
        Optional paths to positive-class example images (few-shot prompting).
    example_class_b_paths : list of str
        Optional paths to negative-class example images (few-shot prompting).
    """

    def __init__(
        self,
        source_dir,
        output_dir,
        target_class_name="Target Object",
        other_class_name="Other Object",
        model="qwen2.5vl:7b",
        example_class_a_paths=None,
        example_class_b_paths=None,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_class_name = target_class_name
        self.other_class_name = other_class_name
        self.model = model

        # Output sub-directories (spaces to underscores, lower-cased)
        self.class_a_dir = self.output_dir / target_class_name.replace(" ", "_").lower()
        self.class_b_dir = self.output_dir / other_class_name.replace(" ", "_").lower()

        # Few-shot examples
        self.example_class_a_paths = example_class_a_paths or []
        self.example_class_b_paths = example_class_b_paths or []

        # Create output directories
        self.class_a_dir.mkdir(parents=True, exist_ok=True)
        self.class_b_dir.mkdir(parents=True, exist_ok=True)

        # Results container
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "target_class": self.target_class_name,
            "other_class": self.other_class_name,
            "total_images": 0,
            "class_a_items": [],
            "class_b_items": [],
            "analysis": {},
            "statistics": {
                "class_a_count": 0,
                "class_b_count": 0,
                "errors": 0,
                "model_calls": 0,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self):
        """
        Return the system prompt.

        CUSTOMISE THIS for your use-case — add visual characteristics,
        counter-examples, or domain-specific language to improve accuracy.
        """
        tgt = self.target_class_name
        oth = self.other_class_name
        return (
            f"You are an expert visual analyst. Your sole task is to decide whether "
            f"the supplied image belongs to the category '{tgt}' or '{oth}'.\n\n"
            f"Focus ONLY on visual features: shape, colour, texture, layout, and context.\n"
            f"Do NOT rely on text or logos in the image.\n\n"
            f"# TODO — add domain-specific visual rules below:\n"
            f"CHARACTERISTICS OF '{tgt.upper()}':\n"
            f"  - [Describe colour / shape / texture ...]\n"
            f"  - [Describe typical context / setting ...]\n\n"
            f"COUNTER-EXAMPLES ('{oth.upper()}'):\n"
            f"  - [Describe what they look like ...]\n"
        )

    def _build_messages(self, image_path):
        """Construct the full chat message list for one image."""
        messages = []
        system_prompt = self._build_system_prompt()

        # few-shot positive examples
        for ex_path in self.example_class_a_paths:
            if os.path.exists(ex_path):
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Is this a '{self.target_class_name}'? Analyse the visual features.",
                        "images": [str(ex_path)],
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "is_target": True,
                            "confidence": 0.95,
                            "reasoning": f"Clearly a {self.target_class_name} due to [visual feature].",
                            "visual_elements": ["element1", "element2"],
                        }),
                    },
                ])

        # few-shot negative examples
        for ex_path in self.example_class_b_paths:
            if os.path.exists(ex_path):
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Is this a '{self.target_class_name}'? Analyse the visual features.",
                        "images": [str(ex_path)],
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "is_target": False,
                            "confidence": 0.90,
                            "reasoning": f"Not a {self.target_class_name}; looks more like a {self.other_class_name}.",
                            "visual_elements": ["element1", "element2"],
                        }),
                    },
                ])

        # actual image to classify
        analysis_prompt = (
            f"{system_prompt}\n\n"
            f"Now analyse the image and respond with ONLY valid JSON:\n"
            f"{{\n"
            f'    "is_target": true or false,\n'
            f'    "confidence": 0.0 to 1.0,\n'
            f'    "reasoning": "one-sentence explanation",\n'
            f'    "visual_elements": ["list", "of", "observed", "features"]\n'
            f"}}\n\n"
            f"Only mark as '{self.target_class_name}' if you are clearly sure."
        )
        messages.append({
            "role": "user",
            "content": analysis_prompt,
            "images": [str(image_path)],
        })
        return messages

    def _call_model(self, image_path):
        """
        Send the image to the Ollama model and parse the JSON response.

        Returns (is_target, confidence, analysis_dict).
        """
        try:
            messages = self._build_messages(image_path)
            response = chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 20,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,
                    "num_predict": 512,
                    "seed": 42,
                },
            )
            self.results["statistics"]["model_calls"] += 1

            raw = response.message.content.strip()

            # Extract the first JSON object found in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                analysis = json.loads(raw[start:end])
            else:
                analysis = json.loads(raw)

            return (
                bool(analysis.get("is_target", False)),
                float(analysis.get("confidence", 0.0)),
                analysis,
            )

        except json.JSONDecodeError:
            # Graceful fallback when the model doesn't return clean JSON
            is_target = "true" in raw.lower() and self.target_class_name.lower() in raw.lower()
            return is_target, 0.5, {
                "is_target": is_target,
                "confidence": 0.5,
                "reasoning": "Could not parse structured response.",
                "raw_response": raw,
            }
        except Exception as exc:
            return False, 0.0, {"error": str(exc)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_images(self, file_pattern=None):
        """
        Classify all images found in source_dir.

        Parameters
        ----------
        file_pattern : str or None
            Optional glob pattern (e.g. "*.png").  When None every
            supported image extension is processed.
        """
        image_files = find_images(self.source_dir, file_pattern)

        if not image_files:
            pattern_hint = file_pattern or f"images ({', '.join(sorted(IMAGE_EXTENSIONS))})"
            print(f"No {pattern_hint} found in: {self.source_dir}")
            return False

        sep = "=" * 80
        print(sep)
        print(f"  ChickenOrEgg  |  {self.target_class_name.upper()}  vs  {self.other_class_name.upper()}")
        print(sep)
        print(f"  Source  : {self.source_dir}")
        print(f"  Output  : {self.output_dir}")
        print(f"  Model   : {self.model}")
        print(f"  Images  : {len(image_files)}")
        if self.example_class_a_paths or self.example_class_b_paths:
            print(f"  Few-shot: +{len(self.example_class_a_paths)} positive, "
                  f"+{len(self.example_class_b_paths)} negative examples")
        print(sep)

        for idx, img_path in enumerate(image_files, 1):
            print(f"  [{idx:>4}/{len(image_files)}] {img_path.name} ...", end=" ", flush=True)

            try:
                is_target, confidence, analysis = self._call_model(img_path)

                record = {
                    "filename": img_path.name,
                    "is_target": is_target,
                    "confidence": confidence,
                    "analysis": analysis,
                }
                self.results["analysis"][img_path.name] = record
                self.results["total_images"] += 1

                if is_target:
                    dest_dir = self.class_a_dir
                    self.results["class_a_items"].append(img_path.name)
                    self.results["statistics"]["class_a_count"] += 1
                    label = f"YES — {self.target_class_name}"
                else:
                    dest_dir = self.class_b_dir
                    self.results["class_b_items"].append(img_path.name)
                    self.results["statistics"]["class_b_count"] += 1
                    label = f"NO  — {self.other_class_name}"

                shutil.copy2(img_path, dest_dir / img_path.name)

                print(f"{label}  (conf={confidence:.2f})")
                print(f"         Reasoning: {analysis.get('reasoning', 'N/A')}")
                elems = analysis.get("visual_elements", [])
                if elems:
                    print(f"         Elements : {', '.join(elems)}")

            except Exception as exc:
                print(f"ERROR — {exc}")
                self.results["statistics"]["errors"] += 1

            time.sleep(0.05)

        return True

    def save_results(self):
        """Write JSON results and a human-readable text report."""
        ts = self.results["timestamp"]
        stats = self.results["statistics"]
        tgt = self.target_class_name
        oth = self.other_class_name

        # JSON
        json_path = self.output_dir / "classification_results.json"
        json_path.write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Text report
        report_path = self.output_dir / "classification_report.txt"
        lines = [
            "=" * 80,
            "ChickenOrEgg Classification Report",
            f"Generated : {ts}",
            f"Model     : {self.model}",
            f"Classes   : '{tgt}'  vs  '{oth}'",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 40,
            f"  Total images   : {self.results['total_images']}",
            f"  {tgt:<34}: {stats['class_a_count']}",
            f"  {oth:<34}: {stats['class_b_count']}",
            f"  Errors          : {stats['errors']}",
        ]

        if self.results["total_images"]:
            success = (
                (self.results["total_images"] - stats["errors"])
                / self.results["total_images"] * 100
            )
            lines.append(f"  Success rate   : {success:.1f}%")

        lines += ["", f"POSITIVE MATCHES — '{tgt}' (sorted by confidence, highest first)", "-" * 60]
        for fname in sorted(
            self.results["class_a_items"],
            key=lambda n: self.results["analysis"][n]["confidence"],
            reverse=True,
        ):
            conf = self.results["analysis"][fname]["confidence"]
            reason = (self.results["analysis"][fname].get("analysis", {}).get("reasoning") or "")[:60]
            lines.append(f"  {fname:<35} conf={conf:.2f}  {reason}")

        lines += ["", f"NEGATIVE MATCHES — '{oth}' (sorted by confidence, highest first)", "-" * 60]
        for fname in sorted(
            self.results["class_b_items"],
            key=lambda n: self.results["analysis"][n]["confidence"],
        ):
            conf = self.results["analysis"][fname]["confidence"]
            reason = (self.results["analysis"][fname].get("analysis", {}).get("reasoning") or "")[:60]
            lines.append(f"  {fname:<35} conf={1-conf:.2f}  {reason}")

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Console summary
        sep = "=" * 80
        print(f"\n{sep}")
        print("  ChickenOrEgg — Done!")
        print(sep)
        print(f"  {tgt}: {stats['class_a_count']}")
        print(f"  {oth}: {stats['class_b_count']}")
        print(f"  Errors      : {stats['errors']}")
        print(f"  Model calls : {stats['model_calls']}")
        print(f"\n  Results : {json_path}")
        print(f"  Report  : {report_path}")
        print(f"  Folders :")
        print(f"    {self.class_a_dir}")
        print(f"    {self.class_b_dir}")
        print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ChickenOrEgg — Generic two-class image classifier powered by Ollama VLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir", required=True,
        help="Directory containing images to classify.",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Directory to save sorted images and reports. (default: ./output)",
    )
    parser.add_argument(
        "--target-class", default="Target Object",
        help="Label for the POSITIVE class  (e.g. 'Pizza').",
    )
    parser.add_argument(
        "--other-class", default="Other Object",
        help="Label for the NEGATIVE class  (e.g. 'Not Pizza').",
    )
    parser.add_argument(
        "--model", default="qwen2.5vl:7b",
        help="Ollama vision model to use. (default: qwen2.5vl:7b)",
    )
    parser.add_argument(
        "--file-pattern", default=None,
        help=(
            "Glob pattern to filter images, e.g. '*.jpg'. "
            "Omit to process ALL supported image types "
            f"({', '.join(sorted(IMAGE_EXTENSIONS))})."
        ),
    )
    parser.add_argument(
        "--example-class-a", nargs="*", default=[],
        metavar="PATH",
        help="Paths to positive-class example images for few-shot prompting.",
    )
    parser.add_argument(
        "--example-class-b", nargs="*", default=[],
        metavar="PATH",
        help="Paths to negative-class example images for few-shot prompting.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"ERROR: source directory not found: {args.source_dir}")
        return

    print("\n" + "=" * 80)
    print("  ChickenOrEgg — Generic Image Classifier (Ollama)")
    print("=" * 80)
    print(f"  Source      : {args.source_dir}")
    print(f"  Output      : {args.output_dir}")
    print(f"  Target class: {args.target_class}")
    print(f"  Other class : {args.other_class}")
    print(f"  Model       : {args.model}")
    print(f"  File pattern: {args.file_pattern or 'all image types'}")
    print()

    if not args.yes:
        answer = input("Start classification? (yes/no): ").strip().lower()
        if answer not in ("yes", "y"):
            print("Cancelled.")
            return

    classifier = GenericVLMClassifier(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_class_name=args.target_class,
        other_class_name=args.other_class,
        model=args.model,
        example_class_a_paths=args.example_class_a,
        example_class_b_paths=args.example_class_b,
    )

    if classifier.process_images(file_pattern=args.file_pattern):
        classifier.save_results()
        print("\nAll done!")
    else:
        print("\nNo images processed. Check --source-dir and --file-pattern.")


if __name__ == "__main__":
    main()
