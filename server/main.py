"""
Vision AI Demo - Real-time Video Analysis Server
High-performance FastAPI server with WebRTC and WebSocket support
"""

import asyncio
import base64
import io
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Set

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

# Import our VLM models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.VLM import VLMProcessor

# Configure logging
import os
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Vision AI Demo")

# Add CORS middleware
allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
)
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

if not allowed_origins:
    allowed_origins = ["*"]

allow_credentials = True
if "*" in allowed_origins:
    allowed_origins = ["*"]
    # Browsers block credentials with wildcard origins, so disable them when '*' is set
    allow_credentials = False

logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vlm_processor: VLMProcessor = None
active_connections: Set[WebSocket] = set()
frame_queue = asyncio.Queue(maxsize=2)  # Small queue to process latest frames only
pcs: Set[RTCPeerConnection] = set()
relay = MediaRelay()

# Processing mode and user input state
current_mode = "caption"  # caption, query, detect, point
user_query_input = None  # For query mode
detect_object_input = None  # For detect/point modes
response_length_setting = "medium"

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window=100):
        self.latencies = deque(maxlen=window)
        self.frame_count = 0
        self.start_time = time.time()

    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
        self.frame_count += 1

    def get_stats(self) -> Dict:
        if not self.latencies:
            return {"status": "no data"}

        runtime = time.time() - self.start_time
        sorted_latencies = sorted(self.latencies)

        return {
            "avg_ms": sum(self.latencies) / len(self.latencies),
            "p50_ms": sorted_latencies[len(sorted_latencies)//2],
            "p95_ms": sorted_latencies[int(len(sorted_latencies)*0.95)],
            "p99_ms": sorted_latencies[int(len(sorted_latencies)*0.99)],
            "fps": self.frame_count / runtime if runtime > 0 else 0,
            "total_frames": self.frame_count
        }

perf_monitor = PerformanceMonitor()

# Frame processor with smart skipping
class FrameProcessor:
    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_frame = None
        self.last_frame_shape = None
        self.motion_threshold = 0.05

    def should_process(self, frame: np.ndarray) -> bool:
        """Decide if frame should be processed based on motion and skip count"""
        self.frame_counter += 1

        # Always process first frame
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.last_frame_shape = frame.shape
            return True

        # Skip frames based on counter
        if self.frame_counter % self.skip_frames != 0:
            return False

        # Check if frame dimensions changed - if so, reset and process
        if frame.shape != self.last_frame_shape:
            self.last_frame = frame.copy()
            self.last_frame_shape = frame.shape
            return True

        try:
            # Motion detection - only process if scene changed significantly
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)

            # Ensure same shape before diff
            if curr_gray.shape != last_gray.shape:
                self.last_frame = frame.copy()
                self.last_frame_shape = frame.shape
                return True

            diff = cv2.absdiff(curr_gray, last_gray)
            change_percent = np.mean(diff) / 255.0

            if change_percent < self.motion_threshold:
                return False

            self.last_frame = frame.copy()
            self.last_frame_shape = frame.shape
            return True

        except Exception as e:
            # On any error, just process the frame
            logger.debug(f"Frame comparison error: {e}")
            self.last_frame = frame.copy()
            self.last_frame_shape = frame.shape
            return True

frame_processor = FrameProcessor(skip_frames=10)  # Skip more frames for faster processing

# Custom video track for WebRTC
class VideoTransformTrack(VideoStreamTrack):
    """Receives video frames and queues them for processing"""

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.counter = 0

    async def recv(self):
        frame = await self.track.recv()
        self.counter += 1

        # Convert to numpy array
        img = frame.to_ndarray(format="rgb24")

        # Queue frame for VLM processing (non-blocking)
        if not frame_queue.full():
            await frame_queue.put(img)
            if self.counter % 30 == 0:  # Log every 30th frame
                logger.info(f"Frame {self.counter} queued for processing. Queue size: {frame_queue.qsize()}")
        else:
            logger.warning(f"Frame queue full! Dropping frame {self.counter}")

        return frame


@app.on_event("startup")
async def startup_event():
    """Initialize VLM model on server startup"""
    global vlm_processor

    logger.info("Starting Vision AI Demo Server...")
    logger.info("Initializing VLM processor...")

    # Get default model from environment or use qwen2vl
    default_model = os.getenv("DEFAULT_MODEL", "qwen2vl")
    vlm_processor = VLMProcessor(model_name=default_model, language="en")
    await vlm_processor.warmup()

    # Start background frame processing task
    asyncio.create_task(process_frames_task())

    logger.info(f"Server ready with {default_model} model! 🚀")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def process_frames_task():
    """Background task that processes frames from queue"""
    logger.info("Frame processing task started")

    global current_mode, user_query_input, detect_object_input, response_length_setting

    while True:
        try:
            # Get frame from queue
            frame = await frame_queue.get()

            # Check if should process
            if not frame_processor.should_process(frame):
                continue

            # Process with VLM using current mode
            start_time = time.time()
            result = await vlm_processor.process_frame(
                frame,
                mode=current_mode,
                user_input=user_query_input or detect_object_input,
                response_length=response_length_setting
            )
            latency_ms = (time.time() - start_time) * 1000

            # Log every frame processing time to identify bottlenecks
            logger.info(f"Frame processed in {latency_ms:.0f}ms (queue size: {frame_queue.qsize()})")

            # Record performance
            perf_monitor.record(latency_ms)

            # Broadcast to all connected clients
            await broadcast_result(result, latency_ms)

            # Log stats every 50 frames
            if perf_monitor.frame_count % 50 == 0:
                stats = perf_monitor.get_stats()
                logger.info(f"Performance: {stats}")

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)


async def broadcast_result(result: Dict, latency_ms: float):
    """Broadcast VLM result to all connected WebSocket clients"""
    if not active_connections:
        return

    # Format message for frontend
    message_data = {
        "type": "caption",
        "timestamp": time.time(),
        "data": {
            "content": result.get("caption", result.get("response", "Processing...")),
            "model": vlm_processor.current_model if vlm_processor else "unknown",
            "confidence": result.get("confidence"),
            "feature": result.get("mode"),
            "latency_ms": round(latency_ms, 2),
            "detections": result.get("detections"),
            "points": result.get("points"),
            "object": result.get("object"),
            "metadata": {
                "fallback_used": result.get("fallback_used", False)
            }
        }
    }

    message = json.dumps(message_data)

    # Send to all clients
    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(connection)

    # Remove disconnected clients
    active_connections.difference_update(disconnected)


@app.get("/")
async def get_index():
    """Serve the client HTML page"""
    client_path = Path(__file__).parent.parent / "client" / "index.html"

    with open(client_path, "r") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for broadcasting captions"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(active_connections)}")

    try:
        # Send initial stats
        stats = perf_monitor.get_stats()
        await websocket.send_json({"type": "stats", "data": stats})

        # Keep connection alive
        while True:
            data = await websocket.receive_text()

            # Handle client messages (e.g., model switching, mode changes, queries)
            try:
                global current_mode, user_query_input, detect_object_input, response_length_setting

                msg = json.loads(data)

                if msg.get("type") == "switch_model":
                    model_name = msg.get("model", "smolvlm")
                    await vlm_processor.switch_model(model_name)

                    # Get supported modes for new model
                    supported_modes = vlm_processor.get_supported_modes()

                    await websocket.send_json({
                        "type": "model_switched",
                        "model": model_name,
                        "supported_modes": supported_modes
                    })

                elif msg.get("type") == "change_mode":
                    # Change processing mode
                    mode = msg.get("mode", "caption")
                    supported_modes = vlm_processor.get_supported_modes()

                    if mode in supported_modes:
                        current_mode = mode
                        await websocket.send_json({
                            "type": "mode_changed",
                            "mode": mode
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Mode {mode} not supported by current model"
                        })

                elif msg.get("type") == "configure":
                    # Handle frontend configuration messages
                    data = msg.get("data", {})
                    query = data.get("query")
                    feature = data.get("feature")
                    response_length = data.get("response_length")

                    if isinstance(response_length, str):
                        normalized_length = response_length.lower()
                        if normalized_length in {"short", "medium", "long"}:
                            response_length_setting = normalized_length
                            logger.info(f"Response length updated: {response_length_setting}")

                    trimmed_query = query.strip() if isinstance(query, str) else None

                    # Update mode based on feature for Moondream
                    if feature:
                        current_mode = feature
                        logger.info(f"Mode updated: {feature}")

                    if feature in {"detection", "point", "mask"}:
                        detect_object_input = trimmed_query
                        user_query_input = None
                        log_msg = trimmed_query or "all objects"
                        logger.info(f"Detection target updated: {log_msg}")
                    else:
                        user_query_input = trimmed_query
                        detect_object_input = None
                        if trimmed_query:
                            logger.info(f"Query updated: {trimmed_query}")
                        else:
                            logger.info("Query cleared")

                elif msg.get("type") == "set_query":
                    # Set user query for query mode
                    user_query_input = msg.get("query", "")
                    await websocket.send_json({
                        "type": "query_set",
                        "query": user_query_input
                    })

                elif msg.get("type") == "set_detect_object":
                    # Set object to detect/point
                    detect_object_input = msg.get("object", "")
                    await websocket.send_json({
                        "type": "detect_object_set",
                        "object": detect_object_input
                    })

                elif msg.get("type") == "get_stats":
                    stats = perf_monitor.get_stats()
                    await websocket.send_json({"type": "stats", "data": stats})

                elif msg.get("type") == "get_model_info":
                    # Get current model capabilities
                    model_stats = vlm_processor.get_stats()
                    supported_modes = vlm_processor.get_supported_modes()

                    await websocket.send_json({
                        "type": "model_info",
                        "model": vlm_processor.current_model,
                        "supported_modes": supported_modes,
                        "stats": model_stats
                    })

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(active_connections)}")


@app.post("/offer")
async def offer(params: dict):
    """Handle WebRTC offer from client"""
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")

        if track.kind == "video":
            # Relay the track and process frames
            local_track = VideoTransformTrack(relay.subscribe(track))
            pc.addTrack(local_track)

    # Handle offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if llama-server is available by checking if port 8080 is accessible
    llamacpp_available = False
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(('127.0.0.1', 8080))
        llamacpp_available = (result == 0)
        sock.close()
    except:
        pass

    # Determine recommended backend based on what's available
    backend_type = "llamacpp" if llamacpp_available else "transformers"

    return {
        "status": "ok",
        "model_loaded": vlm_processor is not None and vlm_processor.model is not None,
        "model": vlm_processor.current_model if vlm_processor else "not loaded",
        "backend_type": backend_type,
        "llamacpp_available": llamacpp_available
    }


@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    stats = perf_monitor.get_stats()
    return {
        "performance": stats,
        "connections": {
            "websocket": len(active_connections),
            "webrtc": len(pcs)
        },
        "queue_size": frame_queue.qsize(),
        "model": vlm_processor.current_model if vlm_processor else "not loaded"
    }


@app.post("/api/process_frame")
async def process_frame(params: dict):
    """Process a single frame via HTTP (simpler than WebRTC)"""
    try:
        global current_mode, user_query_input, detect_object_input, response_length_setting

        # Extract base64 image
        image_data = params.get("image")
        if not image_data:
            return {"error": "No image provided"}

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)

        # Convert RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        req_response_length = params.get("response_length")
        if isinstance(req_response_length, str):
            normalized_length = req_response_length.lower()
            if normalized_length in {"short", "medium", "long"}:
                response_length_setting = normalized_length

        # Get language from request (default: en)
        language = params.get("language", "en")
        if vlm_processor:
            vlm_processor.language = language

        # Process with VLM
        start_time = time.time()
        result = await vlm_processor.process_frame(
            frame,
            mode=current_mode,
            user_input=user_query_input or detect_object_input,
            response_length=response_length_setting
        )
        latency_ms = (time.time() - start_time) * 1000

        # Record performance
        perf_monitor.record(latency_ms)

        return {
            "caption": result.get("caption", result.get("response", "Processing...")),
            "model": vlm_processor.current_model if vlm_processor else "unknown",
            "confidence": result.get("confidence"),
            "feature": result.get("mode"),
            "latency_ms": round(latency_ms, 2),
            "detections": result.get("detections"),
            "points": result.get("points"),
            "object": result.get("object"),
            "fallback_used": result.get("fallback_used", False),
            "language": language
        }

    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/api/process_frame_llamacpp")
async def process_frame_llamacpp(params: dict):
    """Process frame using llama-server backend (FAST - sub-1s)"""
    import aiohttp

    try:
        # Extract base64 image
        image_data = params.get("image")
        if not image_data:
            return {"error": "No image provided"}

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Use prompt from request (frontend sends this)
        prompt = params.get("prompt") or "What objects are visible in this scene?"

        # Get response length preference
        global response_length_setting

        response_length = params.get("response_length", "medium")
        logger.info(f"Response length setting: {response_length}")

        normalized_length = response_length.lower() if isinstance(response_length, str) else "medium"
        if normalized_length in {"short", "medium", "long"}:
            response_length_setting = normalized_length

        # Adjust max_tokens and enhance prompt based on response length
        length_configs = {
            "short": {
                "max_tokens": 50,
                "instruction": "Answer in 1-2 brief sentences only."
            },
            "medium": {
                "max_tokens": 100,
                "instruction": "Provide a balanced description in 2-3 sentences."
            },
            "long": {
                "max_tokens": 200,
                "instruction": "Provide a detailed description with comprehensive observations."
            }
        }

        config = length_configs.get(response_length, length_configs["medium"])
        enhanced_prompt = f"{config['instruction']} {prompt}"

        # Prepare llama-server request
        llama_request = {
            "model": "smolvlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": enhanced_prompt
                        }
                    ]
                }
            ],
            "max_tokens": config["max_tokens"],
            "temperature": 0.0,
            "stream": False,
            "stop": []  # Override default stop tokens to get complete responses
        }

        start_time = time.time()

        # Call llama-server with timeout
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json=llama_request,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"llama-server error: {response.status} - {error_text}")
                    return {"error": f"llama-server error: {response.status}"}

                result = await response.json()
                caption = result["choices"][0]["message"]["content"]

        latency_ms = (time.time() - start_time) * 1000

        # Record performance
        perf_monitor.record(latency_ms)

        logger.info(f"llama.cpp inference: {latency_ms:.1f}ms - Length: {response_length} ({config['max_tokens']} tokens) - Prompt: '{prompt[:30]}...' - {caption[:100]}...")

        return {
            "caption": caption,
            "model": "smolvlm-llamacpp",
            "confidence": None,
            "prompt": prompt,
            "latency_ms": round(latency_ms, 2)
        }

    except asyncio.TimeoutError:
        logger.error("llama-server timeout (>5s)")
        return {"error": "llama-server timeout"}
    except Exception as e:
        logger.error(f"Error calling llama-server: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/api/switch_model")
async def switch_model(params: dict):
    """Switch VLM model"""
    model_name = params.get("model", "smolvlm")

    try:
        await vlm_processor.switch_model(model_name)
        return {"status": "success", "model": model_name}
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,  # Disable reload for performance
        log_level="info",
        access_log=True
    )
