#!/usr/bin/env python
"""
Test frontend integration - simulate what the frontend does
"""

import asyncio
import base64
import io
import json
import aiohttp
from PIL import Image
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8001"


async def test_process_frame_endpoint():
    """Test the /api/process_frame endpoint that the frontend uses"""
    logger.info("=" * 60)
    logger.info("Testing /api/process_frame endpoint")
    logger.info("=" * 60)

    # Create a test image
    test_image = Image.new('RGB', (640, 480), color=(73, 109, 137))

    # Convert to base64 (like frontend does)
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{image_base64}"

    async with aiohttp.ClientSession() as session:
        # Test 1: SmolVLM with query
        logger.info("\n1. Testing SmolVLM with query...")
        payload = {
            "image": image_data
        }

        async with session.post(f"{SERVER_URL}/api/process_frame", json=payload) as response:
            result = await response.json()
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            logger.info(f"Caption: {result.get('caption', 'N/A')}")
            logger.info(f"Model: {result.get('model', 'N/A')}")
            logger.info(f"Latency: {result.get('latency_ms', 'N/A')}ms")

        # Test 2: Switch to Moondream
        logger.info("\n2. Switching to Moondream...")
        switch_payload = {"model": "moondream"}
        async with session.post(f"{SERVER_URL}/api/switch_model", json=switch_payload) as response:
            result = await response.json()
            logger.info(f"Switch result: {json.dumps(result, indent=2)}")

        # Wait a moment for model to load
        await asyncio.sleep(2)

        # Test 3: Process frame with Moondream
        logger.info("\n3. Testing Moondream caption...")
        async with session.post(f"{SERVER_URL}/api/process_frame", json=payload) as response:
            result = await response.json()
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            logger.info(f"Caption: {result.get('caption', 'N/A')}")
            logger.info(f"Model: {result.get('model', 'N/A')}")
            logger.info(f"Latency: {result.get('latency_ms', 'N/A')}ms")

        # Test 4: Switch back to SmolVLM
        logger.info("\n4. Switching back to SmolVLM...")
        switch_payload = {"model": "smolvlm"}
        async with session.post(f"{SERVER_URL}/api/switch_model", json=switch_payload) as response:
            result = await response.json()
            logger.info(f"Switch result: {json.dumps(result, indent=2)}")

        await asyncio.sleep(2)

        # Test 5: Process frame with SmolVLM again
        logger.info("\n5. Testing SmolVLM again...")
        async with session.post(f"{SERVER_URL}/api/process_frame", json=payload) as response:
            result = await response.json()
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            logger.info(f"Caption: {result.get('caption', 'N/A')}")
            logger.info(f"Model: {result.get('model', 'N/A')}")
            logger.info(f"Latency: {result.get('latency_ms', 'N/A')}ms")

        logger.info("\n" + "=" * 60)
        logger.info("Frontend integration test completed successfully! ✅")
        logger.info("=" * 60)


async def main():
    try:
        await test_process_frame_endpoint()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
