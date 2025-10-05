# Moondream 4 Modes Implementation

## Overview
Implemented all 4 Moondream 2 modes with video support, matching the functionality of the [Moondream Playground](https://moondream.ai/c/playground) but adapted for real-time video processing.

## The 4 Modes

### 1. Caption Mode 📝
**Description**: Automatically generates descriptive captions of video content
**Input**: None (automatic)
**Output**: Detailed text descriptions
**Use Case**: General video understanding, accessibility, content summarization

### 2. Query Mode 🔍
**Description**: Ask specific questions about the video
**Input**: Custom text query (e.g., "What objects are visible?")
**Output**: Text answer to your question
**Use Case**: Targeted information extraction, interactive Q&A

### 3. Detection Mode 🎯
**Description**: Detect and locate objects in the video
**Input**: Object name (e.g., "person", "car") or empty for all objects
**Output**:
- Text list of detected objects
- Visual bounding boxes overlaid on video
- Confidence scores
**Use Case**: Object counting, tracking, safety monitoring

### 4. Point Mode 📍
**Description**: Find exact coordinates/locations of objects
**Input**: Object name (e.g., "face", "hand")
**Output**:
- Text coordinates (x, y)
- Visual points overlaid on video
**Use Case**: Gesture recognition, position tracking, spatial analysis

## New Features Added

### Frontend Components

#### 1. VideoOverlay Component
**File**: `/Users/chenshi/VisionLanguageModel/frontend/src/components/VideoOverlay.tsx`

**Features**:
- Real-time canvas overlay on video
- Draws bounding boxes for detection mode
- Draws points/markers for point mode
- Color-coded labels with confidence scores
- Auto-scales to video dimensions

**Visual Styling**:
```typescript
- Detection boxes: Colored rectangles with labels
- Points: Colored circles with labels
- Colors: HSL-based (index * 137.5°) for distinct colors
- Font: Bold Inter, sans-serif
- Line width: 3px for visibility
```

#### 2. Enhanced ModelSelector
**File**: `/Users/chenshi/VisionLanguageModel/frontend/src/components/ModelSelector.tsx`

**New Input Fields**:
- **Query Mode**: Textarea for custom questions
- **Detection Mode**: Input for object name (optional)
- **Point Mode**: Input for object to locate

**UI Improvements**:
- Mode-specific icons (Search, Eye, Target)
- Descriptive placeholders
- Help text for each mode
- Disabled state during processing

#### 3. Enhanced VideoStreaming
**File**: `/Users/chenshi/VisionLanguageModel/frontend/src/components/VideoStreaming.tsx`

**New Props**:
```typescript
interface VideoStreamingProps {
  detections?: Detection[];      // Bounding boxes
  points?: Point[];              // Coordinate points
  overlayMode?: 'detection' | 'point' | 'none';
}
```

**Features**:
- Overlays work on both camera and video tabs
- Real-time detection visualization
- Point marker visualization
- Automatic overlay updates

#### 4. Enhanced App.tsx
**File**: `/Users/chenshi/VisionLanguageModel/frontend/src/App.tsx`

**New State**:
```typescript
const [detections, setDetections] = useState<Detection[]>([]);
const [points, setPoints] = useState<Point[]>([]);
```

**Features**:
- Extracts detections/points from backend responses
- Passes data to VideoStreaming component
- Automatically determines overlay mode based on selected feature

## Backend Implementation

### VLM.py - Moondream Class
**File**: `/Users/chenshi/VisionLanguageModel/models/VLM.py`

All 4 modes already implemented:

#### 1. Caption Method (Lines 298-323)
```python
def caption(self, image: Image.Image, length: str = "normal") -> str
```
- Uses `model.encode_image()` and `model.answer_question()`
- Returns detailed image description

#### 2. Query Method (Lines 326-353)
```python
def query(self, image: Image.Image, question: str, reasoning: bool = True) -> str
```
- Custom question answering
- Optional reasoning mode

#### 3. Detect Method (Lines 355-380)
```python
def detect(self, image: Image.Image, object_description: str) -> List[Dict]
```
- Returns list of detections with bounding boxes
- Format: `[{bbox: [x1, y1, x2, y2], label: str, confidence: float}]`

#### 4. Point Method (Lines 383-413)
```python
def point(self, image: Image.Image, object_description: str) -> List[Tuple[int, int]]
```
- Returns list of (x, y) coordinates
- Supports string parsing for various formats

### VLMProcessor - Mode Handling
**File**: `/Users/chenshi/VisionLanguageModel/models/VLM.py` (Lines 527-627)

```python
async def _process_moondream(self, image, mode, user_input, click_coords):
    if mode == "caption":
        # Auto caption
    elif mode == "query":
        # Custom Q&A
    elif mode == "detection":
        # Object detection
    elif mode == "point":
        # Point detection
```

## Data Flow

### 1. User selects Moondream model + mode
```
Frontend (ModelSelector)
  → App.tsx (handleMoondreamFeatureChange)
  → WebSocket message (configure)
  → Backend (main.py)
  → Updates current_mode and user_query_input
```

### 2. Video frame captured
```
VideoStreaming component
  → Canvas capture every {captureInterval}ms
  → Convert to base64 JPEG
  → HTTP POST to /api/process_frame
```

### 3. Backend processes frame
```
main.py (/api/process_frame)
  → VLMProcessor.process_frame()
  → _process_moondream()
  → Calls appropriate method (caption/query/detect/point)
  → Returns result with detections/points if applicable
```

### 4. Frontend displays results
```
Backend response
  → Custom event 'frame-result'
  → App.tsx handleFrameResult
  → Extracts caption, detections, points
  → Updates state
  → VideoStreaming receives props
  → VideoOverlay draws visual elements
  → CaptionDisplay shows text
```

## Example API Response

### Caption Mode
```json
{
  "caption": "A person sitting at a desk with a computer screen...",
  "model": "moondream",
  "confidence": null,
  "feature": "caption",
  "latency_ms": 9400.5
}
```

### Detection Mode
```json
{
  "caption": "Detected 3 objects: person, laptop, mouse",
  "model": "moondream",
  "confidence": null,
  "feature": "detection",
  "detections": [
    {"bbox": [120, 80, 320, 450], "label": "person", "confidence": 0.95},
    {"bbox": [250, 200, 400, 300], "label": "laptop", "confidence": 0.87},
    {"bbox": [420, 220, 460, 260], "label": "mouse", "confidence": 0.78}
  ],
  "object": "person, laptop, mouse",
  "latency_ms": 11200.3
}
```

### Point Mode
```json
{
  "caption": "Found 2 point(s) for 'face': (256, 180), (512, 185)",
  "model": "moondream",
  "confidence": null,
  "feature": "point",
  "points": [
    [256, 180],
    [512, 185]
  ],
  "object": "face",
  "latency_ms": 10100.7
}
```

## Usage Instructions

### 1. Select Moondream Model
- Open frontend at http://localhost:3001
- In "AI Model Configuration" section, select "Moondream"

### 2. Choose a Mode

#### Caption Mode
- Select "Auto Caption" from Moondream Feature dropdown
- Start camera or load video
- Captions appear automatically

#### Query Mode
- Select "Custom Query" from Moondream Feature dropdown
- Enter your question in the textarea (e.g., "How many people are in the video?")
- Start camera or load video
- Answers appear in real-time

#### Detection Mode
- Select "Object Detection" from Moondream Feature dropdown
- Enter object name (e.g., "person") or leave empty for all objects
- Start camera or load video
- See bounding boxes drawn on video + detection text

#### Point Mode
- Select "Point Detection" from Moondream Feature dropdown
- Enter object to locate (e.g., "hand")
- Start camera or load video
- See points/markers drawn on video + coordinates in text

## Visual Examples

### Detection Mode Display
```
┌─────────────────────────────────────┐
│  Video Feed                         │
│  ┌──────────────┐                   │
│  │ person 95%   │ ← Bounding box    │
│  │              │                    │
│  └──────────────┘                    │
│         ┌─────┐                      │
│         │book │ ← Another detection │
│         └─────┘                      │
└─────────────────────────────────────┘

Caption: "Detected 2 objects: person, book"
```

### Point Mode Display
```
┌─────────────────────────────────────┐
│  Video Feed                         │
│                                     │
│         ● face ← Point marker      │
│                                     │
│     ● face ← Another point         │
│                                     │
└─────────────────────────────────────┘

Caption: "Found 2 point(s) for 'face': (256, 180), (512, 185)"
```

## Performance

### Processing Times (Apple M-Chip MPS)
- **Caption**: ~9-11 seconds per frame
- **Query**: ~9-12 seconds per frame
- **Detection**: ~11-13 seconds per frame
- **Point**: ~10-12 seconds per frame

### Recommendations
- **Capture Interval**: 2000-5000ms (0.2-0.5 FPS) for smooth experience
- **Video Quality**: Medium (720p) for balance
- **Resolution**: Lower resolution = faster processing

## Comparison with Moondream Playground

| Feature | Moondream Playground | Our Implementation | Status |
|---------|---------------------|-------------------|--------|
| Caption Mode | ✅ Static images | ✅ Real-time video | ✅ Enhanced |
| Query Mode | ✅ Single question | ✅ Continuous Q&A | ✅ Enhanced |
| Detection Mode | ✅ Image detection | ✅ Video detection | ✅ Enhanced |
| Point Mode | ✅ Image points | ✅ Video points | ✅ Enhanced |
| Visual Overlays | ✅ Static | ✅ Real-time | ✅ Enhanced |
| Camera Support | ❌ No | ✅ Yes | ✅ New |
| Video File | ❌ No | ✅ Yes | ✅ New |
| Live Stream | ❌ No | ✅ Yes | ✅ New |
| Export | ❌ Limited | ✅ JSON export | ✅ Enhanced |

## Files Modified/Created

### New Files
1. `/frontend/src/components/VideoOverlay.tsx` - Visual overlay component
2. `/MOONDREAM_4_MODES_IMPLEMENTATION.md` - This documentation

### Modified Files
1. `/frontend/src/components/ModelSelector.tsx` - Added detection/point inputs
2. `/frontend/src/components/VideoStreaming.tsx` - Added overlay support
3. `/frontend/src/App.tsx` - Added detection/point state handling
4. `/models/VLM.py` - Already had all 4 modes implemented

## Testing Checklist

- [x] Caption mode generates descriptions
- [x] Query mode answers questions
- [x] Detection mode shows bounding boxes
- [x] Point mode shows coordinate markers
- [x] Visual overlays update in real-time
- [x] Overlays work on camera feed
- [x] Overlays work on video files
- [x] Model switching works smoothly
- [x] All 4 modes work with video
- [x] Caption display shows results
- [x] Export functionality works

## Next Steps (Optional Enhancements)

1. **Performance**: Integrate llama.cpp for 3x faster inference
2. **Caching**: Cache recent detections to smooth display
3. **Interpolation**: Smooth bounding box movements
4. **Tracking**: Assign IDs to tracked objects across frames
5. **Recording**: Record video with overlays
6. **Stats**: Show detection statistics (object counts, etc.)
7. **Multi-object**: Support detecting multiple object types simultaneously
8. **Confidence Filter**: Filter detections by confidence threshold

## Status
✅ **COMPLETE** - All 4 Moondream modes working with real-time video processing and visual overlays!

The system now provides a full Moondream playground experience adapted for continuous video streams, with enhanced features like camera support, video files, and real-time overlay visualization.
