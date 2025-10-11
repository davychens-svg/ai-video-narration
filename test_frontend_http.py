#!/usr/bin/env python3
"""
Test script to simulate frontend calling /api/process_frame
"""
import requests
import base64
from PIL import Image
import io
import numpy as np

# Create a simple test image (gradient pattern like in test_streaming.py)
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

# Convert to base64 data URL (like frontend does)
buffered = io.BytesIO()
image.save(buffered, format="JPEG", quality=80)
img_str = base64.b64encode(buffered.getvalue()).decode()
data_url = f"data:image/jpeg;base64,{img_str}"

print("Testing /api/process_frame endpoint...")
print(f"Image data URL length: {len(data_url)} characters\n")

# Test with transformers backend (default)
print("=" * 60)
print("Test 1: /api/process_frame (transformers backend)")
print("=" * 60)

try:
    response = requests.post(
        "http://localhost:8001/api/process_frame",
        json={
            "image": data_url,
            "prompt": "Describe this image briefly.",
            "response_length": "medium"
        },
        timeout=30
    )

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}\n")
except Exception as e:
    print(f"ERROR: {e}\n")

# Test with llamacpp backend
print("=" * 60)
print("Test 2: /api/process_frame_llamacpp")
print("=" * 60)

try:
    response = requests.post(
        "http://localhost:8001/api/process_frame_llamacpp",
        json={
            "image": data_url,
            "prompt": "Describe this image briefly.",
            "response_length": "medium"
        },
        timeout=30
    )

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}\n")
except Exception as e:
    print(f"ERROR: {e}\n")

print("=" * 60)
print("Test complete!")
print("=" * 60)
