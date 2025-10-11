#!/usr/bin/env python3
"""
Test all three models through HTTP endpoints to verify they're all working
"""
import requests
import base64
from PIL import Image
import io
import numpy as np
import json

def create_test_image():
    """Create a simple gradient test image"""
    width, height = 384, 384
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient
                128                         # Constant blue
            ]
    return Image.fromarray(image_array)

def image_to_data_url(image):
    """Convert PIL image to data URL"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_model(model_name, endpoint, backend='transformers'):
    """Test a specific model"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name.upper()}")
    print(f"Endpoint: {endpoint}")
    print(f"Backend: {backend}")
    print(f"{'='*70}\n")

    image = create_test_image()
    data_url = image_to_data_url(image)

    try:
        response = requests.post(
            f"http://localhost:8001{endpoint}",
            json={
                "image": data_url,
                "prompt": "Describe this image briefly.",
                "response_length": "medium",
                "language": "en"
            },
            timeout=30
        )

        print(f"✅ Status: {response.status_code}")

        if response.ok:
            result = response.json()
            print(f"✅ Caption: {result.get('caption', 'N/A')[:200]}")
            print(f"✅ Model: {result.get('model', 'N/A')}")
            print(f"✅ Latency: {result.get('latency_ms', 'N/A')} ms")
            print(f"\n{'='*70}")
            print("✅ SUCCESS - Model is working!")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("TESTING ALL THREE MODELS")
    print("="*70)
    print("\nThis test verifies that all three models can process frames:")
    print("1. Qwen2-VL (transformers backend)")
    print("2. SmolVLM (llamacpp backend)")
    print("3. Moondream (transformers backend)")
    print("\n" + "="*70 + "\n")

    results = {}

    # Test 1: Qwen2-VL via transformers
    results['qwen2vl'] = test_model(
        'Qwen2-VL',
        '/api/process_frame',
        'transformers'
    )

    # Test 2: SmolVLM via llamacpp
    results['smolvlm'] = test_model(
        'SmolVLM',
        '/api/process_frame_llamacpp',
        'llamacpp'
    )

    # Test 3: Moondream via transformers
    # First, switch to moondream model
    print(f"\n{'='*70}")
    print("Switching to Moondream model...")
    print(f"{'='*70}\n")

    try:
        switch_response = requests.post(
            "http://localhost:8001/api/switch_model",
            json={"model": "moondream"},
            timeout=60
        )

        if switch_response.ok:
            print("✅ Model switched to Moondream")
            import time
            time.sleep(2)  # Give it time to load

            results['moondream'] = test_model(
                'Moondream',
                '/api/process_frame',
                'transformers'
            )
        else:
            print(f"❌ Failed to switch model: {switch_response.status_code}")
            results['moondream'] = False

    except Exception as e:
        print(f"❌ Exception switching model: {e}")
        results['moondream'] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for model, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{model.upper():15} {status}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ ALL MODELS ARE WORKING! Frontend should now show real-time responses.")
    else:
        print("\n❌ SOME MODELS FAILED. Check the errors above.")

    print("\n" + "="*70 + "\n")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
