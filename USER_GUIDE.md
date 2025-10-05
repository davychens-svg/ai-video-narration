# AI Video Narration - User Guide

## Quick Start

### 1. Start the Application

**Backend Server:**
```bash
cd /Users/chenshi/VisionLanguageModel/server
python main.py
```
Server runs on: http://localhost:8001

**Frontend:**
```bash
cd /Users/chenshi/VisionLanguageModel/frontend
npm run dev
```
Frontend runs on: http://localhost:3001

### 2. Open in Browser
Navigate to http://localhost:3001

## Features Overview

### 🎥 Video Input Options

#### Live Camera
1. Click on "Live Camera" tab
2. Click "Start Camera" button
3. Allow camera permissions when prompted
4. Video feed starts automatically

#### Video File/URL
1. Click on "Video File/URL" tab
2. **Option A**: Paste a video URL (MP4, WebM)
   - Or try the sample videos (click "Sample Video 1" or "Sample Video 2")
3. **Option B**: Upload a local video file
   - Click "Choose Video File"
   - Select your video (MP4, WebM, OGG)
4. Click "Load Video" button

### 🤖 AI Models

#### SmolVLM-500M (Fast & Efficient)
- **Speed**: 5-8 seconds per frame
- **Best for**: Real-time captions, quick analysis
- **Features**: Custom queries
- **Memory**: Low usage
- **How to use**:
  1. Select "SmolVLM" from model dropdown
  2. Enter your query (REQUIRED) in the text area
  3. Examples:
     - "What objects are visible?"
     - "Describe the scene"
     - "What colors do you see?"
  4. Click "Send Query" or press Enter

#### Moondream 2 (Advanced & Feature-Rich)
- **Speed**: 9-13 seconds per frame
- **Best for**: Detailed analysis, object detection
- **Features**: 4 different modes (see below)
- **Memory**: Moderate usage
- **How to use**: Select "Moondream" and choose a mode

## Moondream 2 - 4 Modes Explained

### 📝 Mode 1: Auto Caption
**What it does**: Automatically generates detailed descriptions of what's happening in the video

**How to use**:
1. Select "Moondream" model
2. Select "Auto Caption" from Moondream Feature dropdown
3. Start camera or load video
4. Captions appear automatically - no input needed!

**Example Output**:
> "A person sitting at a desk with a laptop. The person is wearing glasses and looking at the screen. There's a coffee mug on the desk and natural light coming from a window on the right."

**Use cases**:
- Video accessibility
- Content summarization
- General video understanding
- Automated logging

---

### 🔍 Mode 2: Custom Query
**What it does**: Ask specific questions about what's happening in the video

**How to use**:
1. Select "Moondream" model
2. Select "Custom Query" from Moondream Feature dropdown
3. Enter your question in the text area
4. Start camera or load video
5. Get answers in real-time!

**Example Queries & Answers**:
- Q: "How many people are in the scene?"
  - A: "There are 2 people visible in the scene."
- Q: "What is the person doing?"
  - A: "The person is typing on a laptop keyboard."
- Q: "What time of day is it?"
  - A: "It appears to be daytime based on the natural lighting."

**Use cases**:
- Interactive Q&A
- Targeted information extraction
- Specific detail inquiry
- Custom analysis

---

### 🎯 Mode 3: Object Detection
**What it does**: Detects and locates objects in the video with bounding boxes

**How to use**:
1. Select "Moondream" model
2. Select "Object Detection" from Moondream Feature dropdown
3. (Optional) Enter object name in the input field:
   - Leave empty to detect ALL objects
   - Or specify: "person", "car", "book", etc.
4. Start camera or load video
5. See colored bounding boxes drawn on the video!

**Visual Display**:
- Colored rectangles around detected objects
- Labels showing object name + confidence %
- Different colors for different objects
- Text output listing all detections

**Example**:
```
Input: "person"
Visual: [Blue box around person] "person 95%"
Text: "Detected 1 object: person"
```

**Use cases**:
- Object counting
- Safety monitoring
- Tracking items
- Inventory management
- Security applications

---

### 📍 Mode 4: Point Detection
**What it does**: Finds exact coordinates/locations of specific objects in the video

**How to use**:
1. Select "Moondream" model
2. Select "Point Detection" from Moondream Feature dropdown
3. Enter object name in the input field (REQUIRED):
   - Examples: "face", "hand", "person", "ball"
4. Start camera or load video
5. See colored markers/dots on the video!

**Visual Display**:
- Colored circles marking object centers
- Labels showing object name
- Coordinates displayed in text
- Different colors for multiple instances

**Example**:
```
Input: "face"
Visual: ● (blue circle on face location)
Text: "Found 2 point(s) for 'face': (256, 180), (512, 185)"
```

**Use cases**:
- Gesture recognition
- Position tracking
- Spatial analysis
- Motion tracking
- Interactive applications

## ⚙️ Settings

Click the settings icon (⚙️) in the Connection Status panel to configure:

### Video Processing

**Video Quality**
- **Low (480p)**: Fastest processing, lower detail
- **Medium (720p)**: ✅ Recommended balance
- **High (1080p)**: Highest detail, slower processing

**Capture Interval**
- Range: 100ms - 5000ms
- Shows equivalent FPS
- **Recommended**: 500ms (2 FPS) for SmolVLM, 2000ms (0.5 FPS) for Moondream
- Lower = faster updates but higher CPU/GPU load
- Higher = slower updates but smoother performance

**Resolution Display**
- Shows actual camera resolution being used
- Auto-adjusts based on selected video quality

## 📊 Real-time Captions Panel

### Features
- Live caption feed with timestamps
- Shows model used (SmolVLM / Moondream)
- Displays feature/mode used
- Shows processing latency in milliseconds
- Auto-scrolls to latest caption

### Actions
- **Export**: Download all captions as JSON file
  - Filename: `captions-YYYY-MM-DD.json`
  - Includes timestamps, model info, and content
- **Clear**: Remove all captions from the panel

## 🔗 Connection Status

Displays 3 status indicators:

1. **WebSocket**: Connection to backend for configuration
   - 🟢 Connected: Live connection
   - 🔴 Disconnected: No connection

2. **Backend API**: Server health status
   - 🟢 Healthy: Server responding
   - 🔴 Error: Server down or unreachable

3. **Video Stream**: Camera/video feed status
   - 🟢 Active: Capturing frames
   - 🔴 Inactive: Not capturing

**Last Heartbeat**: Shows time of last successful communication

**Actions**:
- **Reconnect**: Attempt to reconnect WebSocket
- **Settings**: Open settings dialog

## 💡 Tips & Best Practices

### Performance Optimization

1. **For Real-time Use** (camera):
   - Use SmolVLM model
   - Set capture interval to 500-1000ms
   - Use Medium video quality
   - Close other heavy applications

2. **For Accurate Analysis** (video files):
   - Use Moondream model
   - Set capture interval to 2000-5000ms
   - Use Medium or High video quality
   - Process pre-recorded videos for best results

3. **For Object Detection/Tracking**:
   - Use Moondream Detection/Point modes
   - Increase capture interval to 2000-3000ms
   - Ensure good lighting
   - Stable camera position helps

### Troubleshooting

#### "Model not loaded" errors
- **Cause**: Model is still loading or switching
- **Solution**: Wait 5-10 seconds, errors will stop automatically
- **Prevention**: Don't switch models too frequently

#### No video appears
- **Camera**: Check browser permissions
- **Video file**: Ensure file format is MP4/WebM/OGG
- **Video URL**: Check CORS settings, try sample videos instead

#### Slow processing
- **Solution 1**: Increase capture interval (more ms = slower updates)
- **Solution 2**: Lower video quality
- **Solution 3**: Switch to SmolVLM for faster processing
- **Solution 4**: Close other applications

#### Empty/wrong captions
- **SmolVLM**: Make sure you entered a query (it's required!)
- **Moondream**: Wait for model to fully load (first time takes ~76 seconds)
- **Both**: Check lighting and camera focus

## 🎬 Example Workflows

### Workflow 1: Real-time Room Monitoring
```
1. Select: SmolVLM model
2. Query: "Are there any people in the room?"
3. Video: Start Camera (Live Camera tab)
4. Interval: 1000ms (1 FPS)
5. Result: Real-time updates every second
```

### Workflow 2: Detailed Video Analysis
```
1. Select: Moondream model
2. Mode: Auto Caption
3. Video: Upload your video file
4. Interval: 3000ms (0.33 FPS)
5. Play video and watch detailed captions appear
6. Export captions when done
```

### Workflow 3: Object Counting
```
1. Select: Moondream model
2. Mode: Object Detection
3. Input: "person" (or leave empty for all objects)
4. Video: Start Camera or load video
5. Interval: 2000ms
6. Watch: Bounding boxes appear around objects
7. Read: Text output counts total detections
```

### Workflow 4: Gesture Tracking
```
1. Select: Moondream model
2. Mode: Point Detection
3. Input: "hand"
4. Video: Start Camera
5. Interval: 1000ms
6. Watch: Dots appear marking hand positions
7. Read: Coordinates in text output
```

## 📱 Browser Compatibility

**Recommended Browsers**:
- ✅ Chrome/Chromium (best performance)
- ✅ Edge
- ✅ Safari (may have WebRTC limitations)
- ⚠️ Firefox (works, but may be slower)

**Requirements**:
- WebRTC support for camera
- Canvas API support for rendering
- WebSocket support for real-time updates
- Modern JavaScript (ES2020+)

## 🔐 Privacy & Security

- **Camera**: Video is processed locally, never stored permanently
- **Data**: Frames sent to local server only (localhost:8001)
- **Storage**: Captions stored in browser memory only
- **Export**: JSON export saved locally to your computer
- **Network**: No external API calls (all processing local)

## ⌨️ Keyboard Shortcuts

- **Enter** (in SmolVLM query): Send query
- **Shift+Enter** (in query fields): New line

## 📈 System Architecture

```
┌─────────────┐
│   Browser   │
│  (Frontend) │
└──────┬──────┘
       │ HTTP POST (frames)
       │ WebSocket (config)
       ↓
┌─────────────┐
│   FastAPI   │
│  (Backend)  │
└──────┬──────┘
       │
       │ Process frames
       ↓
┌─────────────┐
│ VLM Models  │
│ SmolVLM /   │
│ Moondream   │
└─────────────┘
```

## 🆘 Support & Issues

For issues or feature requests, please check:
1. Server logs: `/Users/chenshi/VisionLanguageModel/logs/server.log`
2. Browser console (F12)
3. Test with sample videos first
4. Try different models/modes

## 🎓 Learning Resources

- **SmolVLM**: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
- **Moondream**: https://huggingface.co/vikhyatk/moondream2
- **Moondream Playground**: https://moondream.ai/c/playground

---

**Enjoy real-time AI video understanding! 🚀**
