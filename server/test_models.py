#!/usr/bin/env python
"""
Test script to verify both SmolVLM and Moondream models work correctly
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.VLM import VLMProcessor
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_smolvlm():
    """Test SmolVLM model"""
    logger.info("=" * 60)
    logger.info("Testing SmolVLM")
    logger.info("=" * 60)

    processor = VLMProcessor(model_name="smolvlm")

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Test with custom query
    logger.info("Testing SmolVLM with custom query...")
    result = await processor.process_frame(
        dummy_frame,
        mode="caption",
        user_input="What objects are visible in this image?"
    )

    logger.info(f"Result: {result}")
    logger.info(f"Caption: {result.get('caption', 'N/A')}")
    logger.info(f"Model: {result.get('model', 'N/A')}")

    return processor


async def test_moondream():
    """Test Moondream model"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Moondream")
    logger.info("=" * 60)

    processor = VLMProcessor(model_name="moondream")

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Test caption mode
    logger.info("Testing Moondream caption mode...")
    result = await processor.process_frame(
        dummy_frame,
        mode="caption"
    )

    logger.info(f"Result: {result}")
    logger.info(f"Caption: {result.get('caption', 'N/A')}")
    logger.info(f"Model: {result.get('model', 'N/A')}")

    # Test query mode
    logger.info("\nTesting Moondream query mode...")
    result = await processor.process_frame(
        dummy_frame,
        mode="query",
        user_input="What is in this image?"
    )

    logger.info(f"Result: {result}")
    logger.info(f"Caption: {result.get('caption', 'N/A')}")
    logger.info(f"Model: {result.get('model', 'N/A')}")

    return processor


async def test_model_switching():
    """Test switching between models"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Switching")
    logger.info("=" * 60)

    # Start with SmolVLM
    processor = VLMProcessor(model_name="smolvlm")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    logger.info("Starting with SmolVLM...")
    result = await processor.process_frame(
        dummy_frame,
        mode="caption",
        user_input="What is in this image?"
    )
    logger.info(f"SmolVLM result: {result.get('caption', 'N/A')}")

    # Switch to Moondream
    logger.info("\nSwitching to Moondream...")
    await processor.switch_model("moondream")

    result = await processor.process_frame(
        dummy_frame,
        mode="caption"
    )
    logger.info(f"Moondream result: {result.get('caption', 'N/A')}")

    # Switch back to SmolVLM
    logger.info("\nSwitching back to SmolVLM...")
    await processor.switch_model("smolvlm")

    result = await processor.process_frame(
        dummy_frame,
        mode="caption",
        user_input="Describe this image"
    )
    logger.info(f"SmolVLM result: {result.get('caption', 'N/A')}")

    logger.info("\nModel switching test completed successfully!")

    return processor


async def main():
    """Run all tests"""
    try:
        # Test individual models
        await test_smolvlm()
        await test_moondream()

        # Test model switching
        await test_model_switching()

        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully! ✅")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
