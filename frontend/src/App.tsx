import React, { useState, useEffect } from 'react';
import { VideoStreaming } from './components/VideoStreaming';
import { ModelSelector, ModelType, MoondreamFeature } from './components/ModelSelector';
import { CaptionDisplay } from './components/CaptionDisplay';
import { ConnectionStatus } from './components/ConnectionStatus';
import { SettingsDialog } from './components/SettingsDialog';
import { useWebSocket } from './hooks/useWebSocket';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { Toaster, toast } from 'sonner';
import { 
  Eye, 
  Settings, 
  Zap,
  Monitor,
  Smartphone
} from 'lucide-react';

interface Detection {
  bbox: [number, number, number, number];
  label: string;
  confidence?: number;
}

interface Point {
  x: number;
  y: number;
  label?: string;
}

export default function App() {
  // Application state
  const [selectedModel, setSelectedModel] = useState<ModelType>('smolvlm');
  const [moondreamFeature, setMoondreamFeature] = useState<MoondreamFeature>('caption');
  const [customQuery, setCustomQuery] = useState('What objects are visible in this scene?');
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoStreamActive, setVideoStreamActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [points, setPoints] = useState<Point[]>([]);
  
  // Settings state
  const [settings, setSettings] = useState({
    serverUrl: 'http://localhost:8001',
    websocketUrl: 'ws://localhost:8001/ws',
    videoQuality: 'medium' as 'low' | 'medium' | 'high',
    framerate: 15,
    captureInterval: 500, // milliseconds between frame captures
    autoReconnect: true,
    maxRetries: 5,
    debugMode: false
  });

  // WebSocket connection
  const {
    isConnected: wsConnected,
    captions,
    lastError,
    lastHeartbeat,
    connect: connectWS,
    disconnect: disconnectWS,
    sendMessage,
    clearCaptions,
    exportCaptions,
    addCaption
  } = useWebSocket({
    url: settings.websocketUrl,
    autoReconnect: settings.autoReconnect,
    maxRetries: settings.maxRetries,
    debugMode: settings.debugMode
  });

  // Backend connection status (simplified for demo)
  const [backendConnected, setBackendConnected] = useState(false);

  // Check backend connection
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${settings.serverUrl}/health`);
        setBackendConnected(response.ok);
      } catch (error) {
        setBackendConnected(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10 seconds
    
    return () => clearInterval(interval);
  }, [settings.serverUrl]);

  // Handle video stream events
  const handleStreamReady = (stream: MediaStream | null) => {
    setVideoStreamActive(!!stream);
    
    if (stream) {
      toast.success('Video stream started successfully');
      
      // Send model configuration to backend
      sendMessage({
        type: 'configure',
        data: {
          model: selectedModel,
          feature: selectedModel === 'moondream' ? moondreamFeature : null,
          query: customQuery || null, // Send query for both models if provided
          videoQuality: settings.videoQuality,
          framerate: settings.framerate
        }
      });
    } else {
      toast.info('Video stream stopped');
    }
  };

  // Handle model changes
  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);

    if (videoStreamActive) {
      // First, switch the model on the backend
      sendMessage({
        type: 'switch_model',
        model: model
      });

      // Then configure the mode and query
      sendMessage({
        type: 'configure',
        data: {
          model: model,
          feature: model === 'moondream' ? moondreamFeature : null,
          query: customQuery || null // Send query for both models if provided
        }
      });

      toast.info(`Switching to ${model === 'smolvlm' ? 'SmolVLM' : 'Moondream'} model...`);
    }
  };

  const handleMoondreamFeatureChange = (feature: MoondreamFeature) => {
    setMoondreamFeature(feature);
    
    if (videoStreamActive && selectedModel === 'moondream') {
      sendMessage({
        type: 'configure',
        data: {
          model: selectedModel,
          feature: feature,
          query: feature === 'query' ? customQuery : null
        }
      });
      
      toast.info(`Switched to ${feature} feature`);
    }
  };

  const handleCustomQueryChange = (query: string) => {
    setCustomQuery(query);

    // For Moondream, auto-update on change (with debounce)
    // For SmolVLM, only update when Send button is clicked
    if (videoStreamActive && selectedModel === 'moondream') {
      setTimeout(() => {
        sendMessage({
          type: 'configure',
          data: {
            model: selectedModel,
            feature: moondreamFeature,
            query: query || null
          }
        });
      }, 1000);
    }
  };

  const handleSendQuery = () => {
    if (!customQuery.trim() || !videoStreamActive) return;

    sendMessage({
      type: 'configure',
      data: {
        model: selectedModel,
        feature: selectedModel === 'moondream' ? moondreamFeature : null,
        query: customQuery
      }
    });

    toast.success('Query sent to SmolVLM');
  };

  const handleReconnect = () => {
    disconnectWS();
    setTimeout(() => {
      connectWS();
      toast.info('Attempting to reconnect...');
    }, 1000);
  };

  const handleOpenSettings = () => {
    setSettingsOpen(true);
  };

  // Listen for HTTP-based frame results
  useEffect(() => {
    const handleFrameResult = (event: CustomEvent) => {
      const message = event.detail;
      console.log('Received frame result:', message);

      // Add to captions using the WebSocket hook's addCaption method
      if (message.type === 'caption' && message.data) {
        addCaption(message.data);

        // Extract detections if present
        if (message.data.detections && Array.isArray(message.data.detections)) {
          setDetections(message.data.detections);
        } else {
          setDetections([]);
        }

        // Extract points if present
        if (message.data.points && Array.isArray(message.data.points)) {
          const formattedPoints = message.data.points.map((p: any) => ({
            x: p[0] || p.x,
            y: p[1] || p.y,
            label: message.data.object || ''
          }));
          setPoints(formattedPoints);
        } else {
          setPoints([]);
        }
      }
    };

    window.addEventListener('frame-result', handleFrameResult as EventListener);

    return () => {
      window.removeEventListener('frame-result', handleFrameResult as EventListener);
    };
  }, [addCaption]);

  // Show errors as toasts
  useEffect(() => {
    if (lastError) {
      toast.error(lastError);
    }
  }, [lastError]);

  return (
    <div className="min-h-screen animated-bg relative">
      {/* Floating Particles */}
      <div className="particles">
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="particle" />
        ))}
      </div>
      
      <div className="relative z-10">
        <div className="container mx-auto p-6 space-y-8">
          {/* Header */}
          <div className="text-center space-y-6">
            <div className="flex items-center justify-center gap-4">
              <div className="glass p-4 rounded-2xl">
                <Eye className="w-10 h-10 text-foreground drop-shadow-lg" />
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent drop-shadow-lg">
                  AI Video Narration
                </h1>
                <p className="text-foreground/80 text-lg">Real-time video analysis with Vision-Language Models</p>
              </div>
            </div>
            
            <div className="flex items-center justify-center gap-3 flex-wrap">
              <div className="glass px-4 py-2 rounded-full glass-hover">
                <div className="flex items-center gap-2 text-sm">
                  <Zap className="w-4 h-4 text-blue-300" />
                  <span className="text-foreground/90">WebRTC Streaming</span>
                </div>
              </div>
              <div className="glass px-4 py-2 rounded-full glass-hover">
                <div className="flex items-center gap-2 text-sm">
                  <Monitor className="w-4 h-4 text-cyan-300" />
                  <span className="text-foreground/90">Real-time Processing</span>
                </div>
              </div>
              <div className="glass px-4 py-2 rounded-full glass-hover">
                <div className="flex items-center gap-2 text-sm">
                  <Smartphone className="w-4 h-4 text-green-300" />
                  <span className="text-foreground/90">Mobile Ready</span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Video & Model */}
            <div className="lg:col-span-2 space-y-8">
              <div className="glass-card rounded-3xl p-1 glass-hover">
                <VideoStreaming
                  onStreamReady={handleStreamReady}
                  isConnected={wsConnected}
                  captureInterval={settings.captureInterval}
                  videoQuality={settings.videoQuality}
                  serverUrl={settings.serverUrl}
                  detections={detections}
                  points={points}
                  overlayMode={
                    selectedModel === 'moondream' && moondreamFeature === 'detection'
                      ? 'detection'
                      : selectedModel === 'moondream' && moondreamFeature === 'point'
                      ? 'point'
                      : 'none'
                  }
                  backend="llamacpp"
                  prompt={customQuery}
                />
              </div>
              
              <div className="glass-card rounded-3xl p-1 glass-hover">
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelChange={handleModelChange}
                  moondreamFeature={moondreamFeature}
                  onMoondreamFeatureChange={handleMoondreamFeatureChange}
                  customQuery={customQuery}
                  onCustomQueryChange={handleCustomQueryChange}
                  onSendQuery={handleSendQuery}
                  isProcessing={isProcessing}
                />
              </div>
            </div>

            {/* Right Column - Captions & Status */}
            <div className="space-y-8">
              <div className="glass-card rounded-3xl p-1 glass-hover">
                <ConnectionStatus
                  webSocketConnected={wsConnected}
                  backendConnected={backendConnected}
                  videoStreamActive={videoStreamActive}
                  onReconnect={handleReconnect}
                  onOpenSettings={handleOpenSettings}
                  serverUrl={settings.serverUrl}
                  lastHeartbeat={lastHeartbeat || undefined}
                />
              </div>
              
              <div className="glass-card rounded-3xl p-1 glass-hover">
                <CaptionDisplay
                  captions={captions}
                  isConnected={wsConnected}
                  onClearCaptions={clearCaptions}
                  onExportCaptions={exportCaptions}
                />
              </div>
            </div>
          </div>

          {/* Architecture Info */}
          <div className="glass-card rounded-3xl p-8 glass-hover">
            <h2 className="text-2xl font-semibold mb-6 text-center bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
              System Architecture
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-blue-200">Frontend (React)</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                      WebRTC camera capture
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                      YouTube video integration
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                      Real-time WebSocket
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                      Model configuration
                    </li>
                  </ul>
                </div>
              </div>
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-cyan-200">Backend (FastAPI)</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                      Video frame processing
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                      WebRTC handshake
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                      WebSocket broadcasting
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                      Model inference
                    </li>
                  </ul>
                </div>
              </div>
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-green-200">AI Models</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-400"></div>
                      SmolVLM (fast, efficient)
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-400"></div>
                      Moondream (advanced)
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-400"></div>
                      Mac M-Chip & NVIDIA GPU
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-400"></div>
                      Real-time inference
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Dialog */}
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        settings={settings}
        onSettingsChange={setSettings}
      />

      {/* Toast Notifications */}
      <Toaster />
    </div>
  );
}