"""
Newspaper Classifier - Ollama Version using qwen3-vl:4b
Run this script to classify newspaper images using a local AI Vision Language Model.

Requirements:
    pip install ollama pillow
    ollama pull qwen3-vl:4b

Usage:
    python image_dis.py --source-dir /path/to/images --output-dir /path/to/output

This will process all paper*.jpg images in the source directory
and create two folders (newspaper / not_newspaper) with organized results.
"""

import base64
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
from ollama import chat

class NewspaperVLMClassifier:
    def __init__(self, source_dir, output_dir, example_newspaper_paths=None, example_not_newspaper_paths=None):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.newspaper_dir = self.output_dir / "newspaper"
        self.not_newspaper_dir = self.output_dir / "not_newspaper"

        # Few-shot example images (now supporting lists)
        self.example_newspaper_paths = example_newspaper_paths or []
        self.example_not_newspaper_paths = example_not_newspaper_paths or []

        # Create directories
        self.newspaper_dir.mkdir(parents=True, exist_ok=True)
        self.not_newspaper_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "qwen3-vl:4b",
            "total_images": 0,
            "newspapers": [],
            "not_newspapers": [],
            "analysis": {},
            "statistics": {
                "newspaper_count": 0,
                "not_newspaper_count": 0,
                "errors": 0,
                "ollama_calls": 0
            }
        }

    def encode_image_to_base64(self, image_path):
        """
        Convert image to base64 string for API
        """
        try:
            # Open and resize image if too large (max 1024x1024)
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large
                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Save to bytes
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()

                # Encode to base64
                return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")

    def analyze_image_with_vlm(self, image_path):
        """
        Analyze a single image using Ollama qwen3-vl:4b to determine if it's a newspaper
        Returns: (is_newspaper: bool, confidence: float, analysis: dict)
        """
        try:
            # Build messages with few-shot examples if example images are available
            messages = []

            # Add system prompt first
            system_prompt = """You are an expert at identifying newspapers. Focus ONLY on identifying visual layout and color, not text content. Your task is to determine if the image shows a newspaper or not.

NEWSPAPER CHARACTERISTICS:
- Newsprint appearance (black text on white/gray background)
- Small text size with high text density
- Abundant black text with very little colorful elements
- Traditional black and white or grayscale appearance with occasional minimal color accents
- May appear crumpled, folded, or wrinkled - still recognizable as newspaper despite physical condition

KEY VISUAL PATTERNS:
✓ NEWSPAPER: Multiple narrow columns + Dense black text + Headlines + Grid layout + Minimal color usage
✗ NOT NEWSPAPER: Large photos + Minimal text + Single column + Handwritten + Glossy appearance + Heavy color usage + Colorful graphics"""

            # Add multiple newspaper examples if available
            for example_newspaper_path in self.example_newspaper_paths:
                if os.path.exists(example_newspaper_path):
                    messages.extend([
                        {
                            'role': 'user',
                            'content': 'Is this a newspaper? Analyze the visual layout.',
                            'images': [str(example_newspaper_path)]
                        },
                        {
                            'role': 'assistant',
                            'content': '{"is_newspaper": true, "confidence": 0.95, "reasoning": "Clear newspaper layout with masthead, multiple columns, headlines, and dense text formatting", "visual_elements": ["masthead", "multiple columns", "headlines", "dense text", "grid layout"]}'
                        }
                    ])

            # Add multiple non-newspaper examples if available
            for example_not_newspaper_path in self.example_not_newspaper_paths:
                if os.path.exists(example_not_newspaper_path):
                    messages.extend([
                        {
                            'role': 'user',
                            'content': 'Is this a newspaper? Analyze the visual layout.',
                            'images': [str(example_not_newspaper_path)]
                        },
                        {
                            'role': 'assistant',
                            'content': '{"is_newspaper": false, "confidence": 0.90, "reasoning": "Magazine-style layout with large photos and minimal text, lacks newspaper column structure", "visual_elements": ["large photos", "minimal text", "advertisement layout"]}'
                        }
                    ])

            # Add the actual image to analyze
            analysis_prompt = f"""{system_prompt}

Now analyze this image and respond with ONLY a JSON object in this exact format:
{{
    "is_newspaper": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation of your decision",
    "visual_elements": ["list", "of", "key", "elements", "observed"]
}}

Be very strict - only classify as newspaper if it clearly shows newspaper VISUAL LAYOUT and FORMATTING."""

            messages.append({
                'role': 'user',
                'content': analysis_prompt,
                'images': [str(image_path)]
            })

            # Make Ollama call with optimized parameters
            response = chat(
                model='qwen3-vl:4b',
                messages=messages,
                options={
                    'temperature': 0.1,          # Lower temperature for more consistent, focused responses
                    'top_p': 0.9,               # Nucleus sampling for quality control
                    'top_k': 20,                # Limit token choices for more deterministic output
                    'repeat_penalty': 1.1,      # Reduce repetitive text
                    'num_ctx': 4096,            # Optimized context length for image analysis with examples
                    'num_predict': 500,         # Max tokens to generate (enough for JSON response)
                    'seed': 42,                 # Fixed seed for reproducible results
                    # Removed stop tokens to let JSON complete properly
                }
            )

            self.results["statistics"]["ollama_calls"] += 1

            # Parse response
            response_text = response.message.content.strip()

            # Try to extract JSON from response
            try:
                # Find JSON in response (sometimes models add extra text)
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    analysis = json.loads(json_str)
                else:
                    # Fallback parsing
                    analysis = json.loads(response_text)

                is_newspaper = bool(analysis.get("is_newspaper", False))
                confidence = float(analysis.get("confidence", 0.0))

                return is_newspaper, confidence, analysis

            except json.JSONDecodeError:
                # Fallback: try to parse from text
                is_newspaper = "true" in response_text.lower() and "newspaper" in response_text.lower()
                confidence = 0.5  # Default confidence when parsing fails

                return is_newspaper, confidence, {
                    "is_newspaper": is_newspaper,
                    "confidence": confidence,
                    "reasoning": "Failed to parse structured response",
                    "raw_response": response_text
                }

        except Exception as e:
            return False, 0.0, {"error": str(e)}

    def process_all_images(self):
        """Process all images in the source directory"""
        # Find all paper*.jpg files
        image_files = sorted(self.source_dir.glob("paper*.jpg"))

        if len(image_files) == 0:
            print(f"No paper*.jpg files found in {self.source_dir}")
            return False

        print("=" * 80)
        print("NEWSPAPER CLASSIFICATION - VLM PROCESSING")
        print("=" * 80)
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total images found: {len(image_files)}")
        print(f"Model: qwen3-vl:4b")
        print()
        print("Processing images with AI vision model...")
        print("-" * 80)

        for idx, img_path in enumerate(image_files, 1):
            # Progress indicator
            print(f"  Processing {idx}/{len(image_files)}: {img_path.name}...")

            try:
                # Analyze the image
                is_newspaper, confidence, analysis = self.analyze_image_with_vlm(img_path)

                # Record results
                result = {
                    "filename": img_path.name,
                    "is_newspaper": is_newspaper,
                    "confidence": confidence,
                    "vlm_analysis": analysis
                }

                self.results["analysis"][img_path.name] = result
                self.results["total_images"] += 1

                # Copy to appropriate folder
                if is_newspaper:
                    dest_dir = self.newspaper_dir
                    self.results["newspapers"].append(img_path.name)
                    self.results["statistics"]["newspaper_count"] += 1
                    status = "📰 NEWSPAPER"
                else:
                    dest_dir = self.not_newspaper_dir
                    self.results["not_newspapers"].append(img_path.name)
                    self.results["statistics"]["not_newspaper_count"] += 1
                    status = "📄 NOT NEWSPAPER"

                # Copy image
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)

                print(f"    → {status} (confidence: {confidence:.2f})")

                # Display detailed reasoning
                print(f"    💭 Reasoning: {analysis.get('reasoning', 'N/A')}")
                visual_elements = analysis.get('visual_elements', [])
                if visual_elements:
                    print(f"    🔍 Visual elements: {', '.join(visual_elements)}")
                print()

                # Rate limiting: small delay to respect API limits
                time.sleep(0.1)

            except Exception as e:
                print(f"  ⚠ Error processing {img_path.name}: {e}")
                self.results["statistics"]["errors"] += 1

        return True

    def save_results(self):
        """Save results to JSON and generate report"""
        # Save JSON
        results_file = self.output_dir / "vlm_classification_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Generate detailed report
        report_file = self.output_dir / "vlm_classification_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NEWSPAPER CLASSIFICATION - VLM DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {self.results['timestamp']}\n")
            f.write(f"Model Used: {self.results['model']}\n")
            f.write(f"Total Images: {self.results['total_images']}\n")
            f.write(f"Ollama Calls Made: {self.results['statistics']['ollama_calls']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"✓ Newspapers: {self.results['statistics']['newspaper_count']}\n")
            f.write(f"✗ Not Newspapers: {self.results['statistics']['not_newspaper_count']}\n")
            f.write(f"⚠ Errors: {self.results['statistics']['errors']}\n")

            if self.results['total_images'] > 0:
                success_rate = ((self.results['total_images'] - self.results['statistics']['errors'])
                               / self.results['total_images'] * 100)
                f.write(f"Success Rate: {success_rate:.1f}%\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("NEWSPAPERS (sorted by confidence)\n")
            f.write("-" * 80 + "\n")

            newspaper_items = []
            for filename in self.results['newspapers']:
                analysis = self.results['analysis'][filename]
                newspaper_items.append((filename, analysis['confidence']))

            for filename, conf in sorted(newspaper_items, key=lambda x: x[1], reverse=True):
                analysis = self.results['analysis'][filename]
                reasoning = analysis.get('vlm_analysis', {}).get('reasoning', 'N/A')[:50]
                f.write(f"  {filename:30s} confidence: {conf:.2f} | {reasoning}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("NOT NEWSPAPERS (sorted by confidence)\n")
            f.write("-" * 80 + "\n")

            non_newspaper_items = []
            for filename in self.results['not_newspapers']:
                analysis = self.results['analysis'][filename]
                non_newspaper_items.append((filename, analysis['confidence']))

            for filename, conf in sorted(non_newspaper_items, key=lambda x: x[1]):
                analysis = self.results['analysis'][filename]
                reasoning = analysis.get('vlm_analysis', {}).get('reasoning', 'N/A')[:50]
                f.write(f"  {filename:30s} confidence: {1-conf:.2f} | {reasoning}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("NOTES:\n")
            f.write("-" * 80 + "\n")
            f.write("• Images are analyzed by Ollama qwen3-vl:4b model\n")
            f.write("• Images are copied and organized into two folders:\n")
            f.write(f"  - newspaper/ ({self.results['statistics']['newspaper_count']} files)\n")
            f.write(f"  - not_newspaper/ ({self.results['statistics']['not_newspaper_count']} files)\n")
            f.write("• Confidence ranges from 0.0 to 1.0 (higher = more certain)\n")
            f.write("• Classification threshold: 0.5 (>= 0.5 = newspaper)\n")
            f.write("• VLM provides reasoning for each decision\n")
            f.write("• Review detailed analysis in the JSON file for full insights\n")
            f.write("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("VLM PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Total images processed: {self.results['total_images']}")
        print(f"📰 Newspapers: {self.results['statistics']['newspaper_count']}")
        print(f"📄 Not newspapers: {self.results['statistics']['not_newspaper_count']}")
        print(f"⚠  Errors: {self.results['statistics']['errors']}")
        print(f"🤖 Ollama calls made: {self.results['statistics']['ollama_calls']}")
        print()
        print("Results saved to:")
        print(f"  📊 JSON: {results_file}")
        print(f"  📄 Report: {report_file}")
        print()
        print("Organized folders:")
        print(f"  📰 {self.newspaper_dir}")
        print(f"  📄 {self.not_newspaper_dir}")
        print("=" * 80)


def main():
    """Main function"""
    import argparse

    # Resolve default example paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(script_dir, "examples")

    parser = argparse.ArgumentParser(
        description="Classify newspaper images using Ollama qwen3-vl:4b VLM"
    )
    parser.add_argument(
        "--source-dir", required=True,
        help="Directory containing paper*.jpg images to classify"
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Directory to save classification results (default: ./output)"
    )
    parser.add_argument(
        "--example-newspapers", nargs="*",
        default=[
            os.path.join(examples_dir, "paper305.jpg"),
            os.path.join(examples_dir, "paper307.jpg"),
        ],
        help="Paths to example newspaper images for few-shot learning"
    )
    parser.add_argument(
        "--example-not-newspapers", nargs="*",
        default=[
            os.path.join(examples_dir, "paper3.jpg"),
            os.path.join(examples_dir, "paper75.jpg"),
        ],
        help="Paths to example non-newspaper images for few-shot learning"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    SOURCE_DIR = args.source_dir
    OUTPUT_DIR = args.output_dir
    EXAMPLE_NEWSPAPERS = args.example_newspapers
    EXAMPLE_NOT_NEWSPAPERS = args.example_not_newspapers

    print("\n")
    print("=" * 80)
    print("NEWSPAPER IMAGE CLASSIFIER - OLLAMA VERSION")
    print("=" * 80)
    print()
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: qwen3-vl:4b")
    print()

    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return

    # Ask for confirmation unless --yes flag is set
    if not args.yes:
        response = input("Start Ollama processing? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Processing cancelled.")
            return

    print()

    print("📝 Few-shot examples:")
    print(f"  📰 Newspaper examples: {len(EXAMPLE_NEWSPAPERS)}")
    for i, path in enumerate(EXAMPLE_NEWSPAPERS, 1):
        print(f"    {i}. {path}")
    print(f"  📄 Non-newspaper examples: {len(EXAMPLE_NOT_NEWSPAPERS)}")
    for i, path in enumerate(EXAMPLE_NOT_NEWSPAPERS, 1):
        print(f"    {i}. {path}")
    print()

    # Create classifier and process
    classifier = NewspaperVLMClassifier(SOURCE_DIR, OUTPUT_DIR, EXAMPLE_NEWSPAPERS, EXAMPLE_NOT_NEWSPAPERS)

    if classifier.process_all_images():
        classifier.save_results()
        print("\n✓ All done! Check the output directory for Ollama results.")
    else:
        print("\n✗ Processing failed. Please check the source directory.")


if __name__ == "__main__":
    main()