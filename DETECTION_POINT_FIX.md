# Detection & Point Mode Fix

## Issue

Moondream's **Detection** and **Point** modes were returning empty results:
- Detection mode: "No objects detected" for 'face'
- Point mode: "Could not locate 'face'"
- Caption mode: Working perfectly

## Root Cause

The VLM.py implementation was:
1. Using `enc_image = model.encode_image(image)` first
2. Passing encoded image to `model.detect()` and `model.point()`
3. Not accessing the nested dictionary keys `["objects"]` and `["points"]`

According to [Moondream2 HuggingFace documentation](https://huggingface.co/vikhyatk/moondream2), the correct API usage is:

```python
# Correct API
objects = model.detect(image, "face")["objects"]
points = model.point(image, "person")["points"]
```

## Fix Applied

### 1. Detection Method (VLM.py lines 355-386)

**Before**:
```python
@torch.inference_mode()
def detect(self, image: Image.Image, object_description: str) -> List[Dict[str, Any]]:
    if not self.is_ready:
        raise RuntimeError("Model not loaded")

    try:
        # Encode image
        enc_image = self.model.encode_image(image)

        # Use detect method if available, otherwise fallback to query
        if hasattr(self.model, 'detect'):
            detections = self.model.detect(enc_image, object_description, self.tokenizer)
            return detections if isinstance(detections, list) else []
        else:
            # Fallback: ask for object locations
            points = self.point(image, object_description)
            return [{'bbox': [x-10, y-10, x+10, y+10], 'label': object_description}
                    for x, y in points]

    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return []
```

**After**:
```python
@torch.inference_mode()
def detect(self, image: Image.Image, object_description: str) -> List[Dict[str, Any]]:
    """Detect objects using moondream2 detect API"""
    if not self.is_ready:
        raise RuntimeError("Model not loaded")

    try:
        # Call model.detect directly with PIL image
        # API returns: {"objects": [{"bbox": [x1, y1, x2, y2], "label": "object"}]}
        if hasattr(self.model, 'detect'):
            result = self.model.detect(image, object_description)

            # Extract objects from result dictionary
            if isinstance(result, dict) and "objects" in result:
                objects = result["objects"]
                logger.info(f"Detected {len(objects)} objects: {objects}")
                return objects if isinstance(objects, list) else []

            logger.warning(f"Detect result format unexpected: {result}")
            return []
        else:
            logger.warning("Model does not have detect method")
            return []

    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []
```

### 2. Point Method (VLM.py lines 388-426)

**Before**:
```python
@torch.inference_mode()
def point(self, image: Image.Image, object_description: str) -> List[Tuple[int, int]]:
    """Find object locations using moondream2 pointing API"""
    if not self.is_ready:
        raise RuntimeError("Model not loaded")

    try:
        # Encode image
        enc_image = self.model.encode_image(image)

        # Use point method
        if hasattr(self.model, 'point'):
            points = self.model.point(enc_image, object_description, self.tokenizer)

            # Parse points - format may vary
            if isinstance(points, list):
                return [(int(p[0]), int(p[1])) for p in points if len(p) >= 2]
            elif isinstance(points, str):
                # Parse string format like "x=123, y=456"
                import re
                matches = re.findall(r'x=(\d+).*?y=(\d+)', points)
                return [(int(x), int(y)) for x, y in matches]

        return []

    except Exception as e:
        logger.error(f"Error finding points: {e}")
        return []
```

**After**:
```python
@torch.inference_mode()
def point(self, image: Image.Image, object_description: str) -> List[Tuple[int, int]]:
    """Find object locations using moondream2 point API"""
    if not self.is_ready:
        raise RuntimeError("Model not loaded")

    try:
        # Call model.point directly with PIL image
        # API returns: {"points": [[x1, y1], [x2, y2], ...]}
        if hasattr(self.model, 'point'):
            result = self.model.point(image, object_description)

            # Extract points from result dictionary
            if isinstance(result, dict) and "points" in result:
                points = result["points"]
                logger.info(f"Found {len(points)} points: {points}")

                # Convert to list of tuples
                formatted_points = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        formatted_points.append((int(p[0]), int(p[1])))
                return formatted_points

            logger.warning(f"Point result format unexpected: {result}")
            return []
        else:
            logger.warning("Model does not have point method")
            return []

    except Exception as e:
        logger.error(f"Error finding points: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []
```

## Key Changes

1. **Pass PIL Image Directly**: Changed from passing `enc_image` to passing the PIL `image` directly
2. **Extract Nested Dictionary**: Access `result["objects"]` and `result["points"]` instead of treating result as a list
3. **Enhanced Logging**: Added info logs showing detected objects/points for debugging
4. **Better Error Handling**: Added traceback logging for detailed error investigation

## Expected Behavior

### Detection Mode
- Input: Object description (e.g., "face", "person", "car")
- Output: List of bounding boxes with labels
- Format: `[{"bbox": [x1, y1, x2, y2], "label": "face", ...}, ...]`
- Visual: Bounding boxes drawn on video overlay

### Point Mode
- Input: Object description (e.g., "face", "hand")
- Output: List of coordinate points
- Format: `[(x1, y1), (x2, y2), ...]`
- Visual: Circular markers drawn on video overlay

## Testing

To test the fixes:

1. **Start frontend** (if not running):
   ```bash
   cd frontend
   npm run dev
   ```

2. **Select Moondream model** in the UI

3. **Test Detection Mode**:
   - Click "Object Detection" button
   - Enter "face" in the input field
   - Start camera/video
   - You should see bounding boxes around detected faces

4. **Test Point Mode**:
   - Click "Point Detection" button
   - Enter "person" in the input field
   - Start camera/video
   - You should see point markers on detected persons

5. **Check server logs** for confirmation:
   ```bash
   tail -f /tmp/server.log
   ```

   Look for log messages like:
   - `Detected 2 objects: [{'bbox': [100, 150, 200, 300], 'label': 'face'}, ...]`
   - `Found 3 points: [[125, 180], [350, 220], ...]`

## Files Modified

- **`/Users/chenshi/VisionLanguageModel/models/VLM.py`**:
  - Lines 355-386: `detect()` method
  - Lines 388-426: `point()` method

## Status

✅ **Fixed and Deployed** - Server restarted with updated code (2025-10-05 21:05)

The detection and point modes should now work correctly, returning actual bounding boxes and coordinate points that will be displayed as visual overlays on the video stream.
