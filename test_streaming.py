#!/usr/bin/env python3
"""
Test script to demonstrate streaming realtime response for all three models
"""

import asyncio
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.VLM import VLMProcessor


async def test_streaming(model_name: str):
    """Test streaming for a specific model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} with STREAMING")
    print(f"{'='*60}\n")

    # Create processor
    processor = VLMProcessor(model_name=model_name, language="en")

    # Create a simple test image (colorful gradient)
    width, height = 384, 384
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient
                128                         # Constant blue
            ]

    # Convert to PIL Image
    image = Image.fromarray(image_array)

    print("Generated test image (gradient pattern)")
    print("\nGenerating caption with STREAMING:\n")

    # Test streaming caption
    start_time = time.time()
    full_text = ""

    # Get the model's caption method directly to pass stream parameter
    if hasattr(processor.model, 'caption'):
        result_generator = processor.model.caption(image, stream=True)

        # Iterate through streaming chunks
        chunk_count = 0
        for chunk in result_generator:
            chunk_count += 1
            full_text += chunk
            print(f"[Chunk {chunk_count}] {chunk}", end='', flush=True)

            # Simulate realtime display with small delay
            await asyncio.sleep(0.05)

        elapsed = time.time() - start_time

        print(f"\n\n{'='*60}")
        print(f"STREAMING COMPLETE")
        print(f"{'='*60}")
        print(f"Total chunks: {chunk_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Full caption: {full_text}")
        print(f"{'='*60}\n")
    else:
        print(f"❌ Model {model_name} doesn't support streaming caption method")


async def test_non_streaming(model_name: str):
    """Test non-streaming for comparison"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} WITHOUT streaming (baseline)")
    print(f"{'='*60}\n")

    # Create processor
    processor = VLMProcessor(model_name=model_name, language="en")

    # Create the same test image
    width, height = 384, 384
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128
            ]

    image = Image.fromarray(image_array)

    print("Generating caption WITHOUT streaming (waiting for complete response)...")
    print("...", flush=True)

    # Test non-streaming caption
    start_time = time.time()

    if hasattr(processor.model, 'caption'):
        result = processor.model.caption(image, stream=False)
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"NON-STREAMING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Full caption: {result}")
        print(f"{'='*60}\n")
    else:
        print(f"❌ Model {model_name} doesn't have caption method")


async def main():
    """Run all tests"""
    models = ["smolvlm", "moondream", "qwen2vl"]

    print("\n" + "="*60)
    print("STREAMING VS NON-STREAMING COMPARISON TEST")
    print("="*60)
    print("\nThis test demonstrates the difference between:")
    print("1. STREAMING: Tokens appear progressively (realtime response)")
    print("2. NON-STREAMING: Complete response after generation finishes")
    print("\n" + "="*60 + "\n")

    for model in models:
        try:
            # Test streaming
            await test_streaming(model)

            # Test non-streaming for comparison
            await test_non_streaming(model)

            print("\n" + "~"*60 + "\n")

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error testing {model}: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next model...\n")
            continue

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
