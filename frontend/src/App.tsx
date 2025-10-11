import { useState, useEffect, useCallback, useRef } from 'react';
import { VideoStreaming } from './components/VideoStreaming';
import { ModelSelector, ModelType, MoondreamFeature } from './components/ModelSelector';
import { CaptionDisplay } from './components/CaptionDisplay';
import { ConnectionStatus } from './components/ConnectionStatus';
import { SettingsDialog } from './components/SettingsDialog';
import { useWebSocket } from './hooks/useWebSocket';
import { Toaster, toast } from 'sonner';
import { Eye } from 'lucide-react';
import { LanguageToggle } from './components/LanguageToggle';
import { Language, getTranslations } from './lib/i18n';

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
  const [language, setLanguage] = useState<Language>('en');
  const translations = getTranslations(language);
  const resolveDefaultConnections = () => {
    const envServerUrl = import.meta.env.VITE_SERVER_URL as string | undefined;
    const envWebsocketUrl = import.meta.env.VITE_WS_URL as string | undefined;

    if (envServerUrl && envWebsocketUrl) {
      return {
        serverUrl: envServerUrl,
        websocketUrl: envWebsocketUrl
      };
    }

    if (typeof window !== 'undefined') {
      const { protocol, hostname, port } = window.location;
      const isLocalHost = hostname === 'localhost' || hostname === '127.0.0.1';

      let backendPort: string | null = null;
      if (isLocalHost) {
        backendPort = '8001';
      } else if (protocol === 'http:' && port && port !== '' && !['80', '443'].includes(port)) {
        backendPort = port;
      }

      const serverUrl = `${protocol}//${hostname}${backendPort ? `:${backendPort}` : ''}`;
      const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
      const websocketUrl = `${wsProtocol}//${hostname}${backendPort ? `:${backendPort}` : ''}/ws`;

      return { serverUrl, websocketUrl };
    }

    return {
      serverUrl: 'http://localhost:8001',
      websocketUrl: 'ws://localhost:8001/ws'
    };
  };

  // Application state
  const [selectedModel, setSelectedModel] = useState<ModelType>('qwen2vl');
  const [moondreamFeature, setMoondreamFeature] = useState<MoondreamFeature>('caption');
  const [customQuery, setCustomQuery] = useState(translations.defaultPrompt);
  const [isModelSwitching, setIsModelSwitching] = useState(false);
  const [videoStreamActive, setVideoStreamActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [points, setPoints] = useState<Point[]>([]);
  const defaultPromptRef = useRef(translations.defaultPrompt);

  useEffect(() => {
    const newPrompt = getTranslations(language).defaultPrompt;
    if (customQuery === defaultPromptRef.current) {
      setCustomQuery(newPrompt);
    }
    defaultPromptRef.current = newPrompt;
  }, [language]);

  // Settings state
  const [settings, setSettings] = useState(() => {
    const defaults = resolveDefaultConnections();

    return {
      serverUrl: defaults.serverUrl,
      websocketUrl: defaults.websocketUrl,
      videoQuality: 'medium' as 'low' | 'medium' | 'high',
      framerate: 15,
      captureInterval: 500, // milliseconds between frame captures
      autoReconnect: true,
      maxRetries: 5,
      debugMode: false,
      responseLength: 'medium' as 'short' | 'medium' | 'long'
    };
  });

  // WebSocket connection
  const applySceneData = useCallback((payload: any) => {
    if (payload?.detections && Array.isArray(payload.detections)) {
      setDetections(payload.detections);
    } else {
      setDetections([]);
    }

    if (payload?.points && Array.isArray(payload.points)) {
      const labelBase = payload?.object ? String(payload.object) : '';
      const labelSuffix = payload?.metadata?.fallback_used ? ' (est)' : '';
      const formattedPoints = payload.points.map((p: any) => ({
        x: p[0] ?? p.x,
        y: p[1] ?? p.y,
        label: labelBase ? `${labelBase}${labelSuffix}` : undefined
      }));
      setPoints(formattedPoints);
    } else {
      setPoints([]);
    }
  }, []);

  const {
    isConnected: wsConnected,
    captions,
    lastError,
    sendMessage,
    clearCaptions,
    exportCaptions,
    addCaption,
    currentModel
  } = useWebSocket({
    url: settings.websocketUrl,
    autoReconnect: settings.autoReconnect,
    maxRetries: settings.maxRetries,
    debugMode: settings.debugMode,
    onCaptionData: applySceneData
  });

  useEffect(() => {
    setIsModelSwitching(currentModel !== selectedModel);
  }, [currentModel, selectedModel]);

  const modelReady = currentModel === selectedModel && !isModelSwitching;

  // Backend connection status and type detection
  const [backendConnected, setBackendConnected] = useState(false);
  const [backendType, setBackendType] = useState<'llamacpp' | 'transformers'>('transformers');

  // Check backend connection and detect backend type
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${settings.serverUrl}/health`);
        if (response.ok) {
          const data = await response.json();
          setBackendConnected(true);
          // Use backend_type from health check to determine which endpoint to use
          setBackendType(data.backend_type || 'transformers');
        } else {
          setBackendConnected(false);
        }
      } catch (error) {
        setBackendConnected(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, [settings.serverUrl]);

  const normalizeQuery = useCallback((value: string) => {
    const trimmed = value?.trim() ?? '';
    return trimmed.length > 0 ? trimmed : null;
  }, []);

  const pushConfiguration = useCallback(
    (
      model: ModelType,
      feature: MoondreamFeature | null,
      queryValue: string | null,
      successMessage: string | undefined = undefined,
      extraData: Record<string, unknown> = {}
    ) => {
      if (!wsConnected) {
        return false;
      }

      sendMessage({
        type: 'configure',
        data: {
          model,
          feature,
          query: queryValue,
          response_length: settings.responseLength,
          ...extraData
        }
      });

      if (successMessage) {
        toast.success(successMessage);
      }

      return true;
    },
    [wsConnected, sendMessage, settings.responseLength]
  );

  // Handle video stream events
  const handleStreamReady = (stream: MediaStream | null) => {
    setVideoStreamActive(!!stream);
    
    if (stream) {
      toast.success(translations.toastStreamStarted);

      if (currentModel !== selectedModel) {
        sendMessage({
          type: 'switch_model',
          model: selectedModel
        });
      }

      // Send model configuration to backend
      const feature = selectedModel === 'moondream' ? moondreamFeature : null;
      const queryValue =
        selectedModel === 'moondream'
          ? (feature === 'caption' ? null : normalizeQuery(customQuery))
          : normalizeQuery(customQuery);  // Qwen2-VL and SmolVLM both require queries

      pushConfiguration(selectedModel, feature, queryValue, undefined, {
        videoQuality: settings.videoQuality,
        framerate: settings.framerate
      });
    } else {
      toast.info(translations.toastStreamStopped);
    }
  };

  // Handle model changes
  const handleModelChange = (model: ModelType) => {
    if (selectedModel === model) {
      return;
    }

    setSelectedModel(model);

    if (wsConnected && currentModel !== model) {
      sendMessage({
        type: 'switch_model',
        model
      });
    }

    // Configure mode/query regardless of switching state
    const feature = model === 'moondream' ? moondreamFeature : null;
    const queryValue =
      model === 'moondream'
        ? (feature === 'caption' ? null : normalizeQuery(customQuery))
        : normalizeQuery(customQuery);
    pushConfiguration(
      model,
      feature,
      queryValue,
      undefined,
      model === 'moondream'
        ? { videoQuality: settings.videoQuality, framerate: settings.framerate }
        : {}
    );

    if (videoStreamActive) {
      const modelLabel = model === 'qwen2vl' ? translations.modelQwen2VL : model === 'smolvlm' ? translations.modelSmolVLM : translations.modelMoondream;
      toast.info(translations.toastModelSwitching.replace('{model}', modelLabel));
    }
  };

  const handleMoondreamFeatureChange = (feature: MoondreamFeature) => {
    setMoondreamFeature(feature);
    
    if (selectedModel === 'moondream') {
      const queryValue =
        feature === 'caption'
          ? null
          : normalizeQuery(customQuery);

      pushConfiguration('moondream', feature, queryValue);
      const featureLabel = translations.moondreamFeatureNames[feature];
      toast.info(translations.toastFeatureSwitched.replace('{feature}', featureLabel));
    }
  };

  const handleCustomQueryChange = (query: string) => {
    setCustomQuery(query);
  };

  const handleSendQuery = () => {
    const trimmedQuery = normalizeQuery(customQuery);
    if (!trimmedQuery || !videoStreamActive) return;

    const modelLabel = selectedModel === 'qwen2vl' ? translations.modelQwen2VL : selectedModel === 'smolvlm' ? translations.modelSmolVLM : translations.modelMoondream;
    const success = pushConfiguration(
      selectedModel,
      selectedModel === 'moondream' ? moondreamFeature : null,
      trimmedQuery,
      translations.toastQuerySent.replace('{model}', modelLabel)
    );

    if (!success) {
      toast.error(translations.toastWsError);
    }
  };

  const handleSendMoondreamSettings = () => {
    if (selectedModel !== 'moondream') {
      return;
    }

    const feature = moondreamFeature;
    const queryValue = feature === 'caption' ? null : normalizeQuery(customQuery);
    const sent = pushConfiguration('moondream', feature, queryValue, translations.toastMoondreamUpdated);

    if (!sent) {
      toast.error(translations.toastWsError);
    }
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
        applySceneData(message.data);
      }
    };

    window.addEventListener('frame-result', handleFrameResult as EventListener);

    return () => {
      window.removeEventListener('frame-result', handleFrameResult as EventListener);
    };
  }, [addCaption, applySceneData]);

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
                  {translations.title}
                </h1>
                <p className="text-foreground/80 text-lg">{translations.subtitle}</p>
              </div>
            </div>
            <div className="flex justify-center">
              <LanguageToggle language={language} onLanguageChange={setLanguage} />
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
                      : selectedModel === 'moondream' && moondreamFeature === 'mask'
                      ? 'mask'
                      : 'none'
                  }
                  backend={
                    selectedModel === 'moondream' || selectedModel === 'qwen2vl'
                      ? 'transformers'  // Moondream and Qwen2-VL always use transformers
                      : backendType     // SmolVLM uses detected backend (llamacpp on Mac, transformers on Linux)
                  }
                  prompt={customQuery}
                  modelReady={modelReady}
                  responseLength={settings.responseLength}
                  language={language}
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
                  onMoondreamSubmit={handleSendMoondreamSettings}
                  isProcessing={isModelSwitching}
                  responseLength={settings.responseLength}
                  language={language}
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
                  onOpenSettings={handleOpenSettings}
                  serverUrl={settings.serverUrl}
                  language={language}
                />
              </div>
              
              <div className="glass-card rounded-3xl p-1 glass-hover">
                <CaptionDisplay
                  captions={captions}
                  isConnected={wsConnected}
                  onClearCaptions={clearCaptions}
                  onExportCaptions={exportCaptions}
                  language={language}
                />
              </div>
            </div>
          </div>

          {/* Architecture Info */}
          <div className="glass-card rounded-3xl p-8 glass-hover">
            <h2 className="text-2xl font-semibold mb-6 text-center bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
              {translations.architectureTitle}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-blue-200">{translations.architectureFrontendTitle}</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    {translations.architectureFrontendItems.map((item) => (
                      <li key={item} className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400"></div>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-cyan-200">{translations.architectureBackendTitle}</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    {translations.architectureBackendItems.map((item) => (
                      <li key={item} className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="space-y-4">
                <div className="glass p-4 rounded-2xl">
                  <h4 className="font-semibold text-lg mb-3 text-green-200">{translations.architectureModelsTitle}</h4>
                  <ul className="space-y-2 text-sm text-foreground/80">
                    {translations.architectureModelsItems.map((item) => (
                      <li key={item} className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-green-400"></div>
                        {item}
                      </li>
                    ))}
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
        language={language}
      />

      {/* Toast Notifications */}
      <Toaster />
    </div>
  );
}
