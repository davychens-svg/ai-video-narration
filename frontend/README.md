# Vision AI Demo - React Frontend

Modern, glassmorphic UI for real-time video analysis powered by Vision-Language Models.

## Features

✨ **Glassmorphic Design** - Modern frosted glass aesthetic with animated gradients
🎥 **WebRTC Streaming** - Real-time video capture from camera
🔌 **WebSocket Integration** - Live caption broadcasting
🤖 **Dual Model Support** - SmolVLM (fast) & Moondream 3.0 (feature-rich)
📱 **Responsive** - Works on desktop and mobile
⚡ **Fast Performance** - Vite + React + TypeScript

## Tech Stack

- **Framework**: React 18.3 + TypeScript
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS 3.4 + Custom Glassmorphism
- **UI Components**: shadcn/ui + Radix UI
- **Icons**: Lucide React
- **Notifications**: Sonner
- **State Management**: React Hooks

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── VideoStreaming.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── CaptionDisplay.tsx
│   │   ├── ConnectionStatus.tsx
│   │   └── SettingsDialog.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts  # WebSocket hook
│   ├── lib/
│   │   └── utils.ts         # Utility functions
│   ├── App.tsx              # Main app component
│   ├── main.tsx             # Entry point
│   └── globals.css          # Global styles
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend server running on port 8001

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start on http://localhost:3001 (or 3000 if available)

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## Configuration

### Backend Endpoints

The frontend connects to these backend endpoints (configured in `App.tsx`):

- **HTTP API**: `http://localhost:8001`
- **WebSocket**: `ws://localhost:8001/ws`
- **WebRTC Offer**: `http://localhost:8001/offer`

### Vite Proxy

The `vite.config.ts` configures proxy for development:

```typescript
server: {
  port: 3000,
  proxy: {
    '/ws': 'ws://localhost:8001',
    '/offer': 'http://localhost:8001',
    '/api': 'http://localhost:8001',
  }
}
```

## Components

### VideoStreaming
- WebRTC camera capture
- Real-time video display
- Stream management (start/stop)

### ModelSelector
- Model switching (SmolVLM / Moondream)
- Mode selection for Moondream (Caption, Query, Detect, Point)
- Custom query input
- Real-time configuration

### CaptionDisplay
- Live caption feed
- Auto-scroll
- Export/clear functionality
- Timestamp display

### ConnectionStatus
- WebSocket connection indicator
- Backend health check
- Video stream status
- Reconnection controls

### SettingsDialog
- Server URL configuration
- Video quality settings
- Framerate control
- Auto-reconnect options
- Debug mode

## Models & Features

### SmolVLM (Fast)
- **Mode**: Caption only
- **Speed**: ~100-300ms per frame
- **Use Case**: Real-time narration

### Moondream 3.0 (Advanced)
- **Caption**: Detailed scene descriptions
- **Query**: Ask questions about the video
- **Detect**: Object detection with bounding boxes
- **Point**: Find object locations as coordinates

## Styling

### Glassmorphism

Custom CSS variables in `globals.css`:

```css
--glass-bg: rgba(255, 255, 255, 0.1);
--glass-border: rgba(255, 255, 255, 0.2);
--glass-shadow: rgba(0, 0, 0, 0.1);
```

### Animated Background

Gradient animation with floating particles:

```css
.animated-bg {
  background: linear-gradient(135deg, #0ea5e9, #06b6d4, #10b981);
  animation: gradient 15s ease infinite;
}
```

## Development

### Hot Reload

Vite provides instant HMR (Hot Module Replacement):

```bash
npm run dev
```

### Type Checking

```bash
# Run TypeScript type checking
npm run build
```

### Linting

```bash
npm run lint
```

## WebRTC Flow

1. User clicks "Start" button
2. Browser requests camera permission
3. `getUserMedia()` captures video stream
4. WebRTC peer connection established with backend
5. Video frames sent to backend via RTC Data Channel
6. Backend processes frames with VLM
7. Captions broadcast via WebSocket
8. UI updates in real-time

## WebSocket Events

### Outgoing (Frontend → Backend)

```typescript
// Configure model
{
  type: 'configure',
  data: {
    model: 'smolvlm' | 'moondream',
    feature: 'caption' | 'query' | 'detect' | 'point',
    query?: string,
    videoQuality: 'low' | 'medium' | 'high',
    framerate: number
  }
}
```

### Incoming (Backend → Frontend)

```typescript
// Caption result
{
  type: 'caption',
  data: string,
  latency_ms: number,
  timestamp: number
}

// Query result
{
  type: 'query',
  question: string,
  data: string,
  latency_ms: number
}

// Detection result
{
  type: 'detect',
  data: Array<{bbox: number[], label: string}>,
  latency_ms: number
}
```

## Troubleshooting

### Port Already in Use

If port 3000/3001 is occupied:

```bash
# Vite automatically tries the next available port
# Or specify a custom port:
npm run dev -- --port 3002
```

### WebSocket Connection Failed

1. Check backend is running on port 8001
2. Verify WebSocket URL in settings
3. Check browser console for errors
4. Try manual reconnect

### Camera Access Denied

1. Check browser permissions
2. Ensure HTTPS (required for WebRTC on remote hosts)
3. Try different browser

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

## Performance

- **Initial Load**: < 2s (optimized bundle splitting)
- **HMR**: < 100ms (Vite instant updates)
- **Frame Processing**: Depends on backend model
  - SmolVLM: ~100-300ms
  - Moondream: ~500-2000ms

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers with WebRTC support

## License

