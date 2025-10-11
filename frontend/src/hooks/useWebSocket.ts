import { useState, useEffect, useRef, useCallback } from 'react';
import { Caption } from '../components/CaptionDisplay';

interface UseWebSocketProps {
  url: string;
  autoReconnect?: boolean;
  maxRetries?: number;
  debugMode?: boolean;
  onCaptionData?: (data: any) => void;
}

type WebSocketMessage = {
  type: string;
  timestamp?: string;
  [key: string]: any;
};

export function useWebSocket({ 
  url, 
  autoReconnect = true, 
  maxRetries = 5,
  debugMode = false,
  onCaptionData
}: UseWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [lastHeartbeat, setLastHeartbeat] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string>('smolvlm');
  const [supportedModes, setSupportedModes] = useState<string[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const log = useCallback((message: string) => {
    if (debugMode) {
      console.log(`[WebSocket] ${message}`);
    }
  }, [debugMode]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      log('Already connected');
      return;
    }

    try {
      log(`Connecting to ${url}`);
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        log('Connected successfully');
        setIsConnected(true);
        setLastError(null);
        setRetryCount(0);
        setLastHeartbeat(new Date().toISOString());

        // Start heartbeat
        heartbeatIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
            setLastHeartbeat(new Date().toISOString());
          }
        }, 30000);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          log(`Received message: ${message.type}`);

          switch (message.type) {
            case 'caption': {
              const captionContent =
                message.data?.caption ??
                message.data?.content ??
                message.data?.error ??
                'No response';

              const newCaption: Caption = {
                id: Date.now().toString(),
                timestamp: message.timestamp || new Date().toISOString(),
                content: captionContent,
                model: message.data?.model ?? 'unknown',
                confidence: message.data?.confidence,
                feature: message.data?.feature,
                latency_ms: message.data?.latency_ms
              };

              setCaptions(prev => [...prev, newCaption]);

              if (onCaptionData) {
                onCaptionData(message.data);
              }
              break;
            }
            
            case 'status':
              log(`Status update: ${JSON.stringify(message.data)}`);
              setLastHeartbeat(new Date().toISOString());
              break;

            case 'model_switched':
              log(`Model switched to ${message.model}`);
              setCurrentModel(message.model);
              if (Array.isArray(message.supported_modes)) {
                setSupportedModes(message.supported_modes);
              }
              break;

            case 'mode_changed':
              log(`Mode changed to ${message.mode}`);
              break;
            
            case 'error':
              // Payload may be {data: {message}} or {message: "..."}
              const errorMessage =
                (message.data && (message.data.message || message.data.error)) ||
                message.message ||
                'Server reported an unknown error.';
              log(`Server error: ${errorMessage}`);
              setLastError(errorMessage);
              if (typeof message.active_model === 'string') {
                setCurrentModel(message.active_model);
              }
              break;
            
            default:
              log(`Unknown message type: ${message.type}`);
          }
        } catch (error) {
          log(`Failed to parse message: ${error}`);
        }
      };

      wsRef.current.onclose = (event) => {
        log(`Connection closed: ${event.code} - ${event.reason}`);
        setIsConnected(false);
        
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
          heartbeatIntervalRef.current = null;
        }

        if (autoReconnect && retryCount < maxRetries && event.code !== 1000) {
          const timeout = Math.min(1000 * Math.pow(2, retryCount), 30000);
          log(`Reconnecting in ${timeout}ms (attempt ${retryCount + 1}/${maxRetries})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setRetryCount(prev => prev + 1);
            connect();
          }, timeout);
        }
      };

      wsRef.current.onerror = (error) => {
        log(`Connection error: ${error}`);
        setLastError('WebSocket connection failed');
      };

    } catch (error) {
      log(`Failed to create WebSocket: ${error}`);
      setLastError('Failed to create WebSocket connection');
    }
  }, [url, autoReconnect, maxRetries, retryCount, log, onCaptionData]);

  const disconnect = useCallback(() => {
    log('Disconnecting...');
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setRetryCount(0);
  }, [log]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      log(`Sent message: ${JSON.stringify(message)}`);
      return true;
    } else {
      log('Cannot send message: WebSocket not connected');
      return false;
    }
  }, [log]);

  const clearCaptions = useCallback(() => {
    setCaptions([]);
  }, []);

  const exportCaptions = useCallback(() => {
    const data = JSON.stringify(captions, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `captions-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [captions]);

  const addCaption = useCallback((captionData: any) => {
    const newCaption: Caption = {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date().toISOString(),
      content: captionData.caption || captionData.content || captionData.error || 'No response',
      model: captionData.model || 'unknown',
      confidence: captionData.confidence,
      feature: captionData.feature,
      latency_ms: captionData.latency_ms
    };
    setCaptions(prev => [...prev, newCaption]);
    log(`Added caption from HTTP: ${newCaption.content.substring(0, 50)}...`);
  }, [log]);

  // Auto-connect on mount
  useEffect(() => {
    if (url) {
      connect();
    }

    return () => {
      disconnect();
    };
    // We intentionally avoid depending on `connect`/`disconnect` to prevent rapid re-connect loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    captions,
    lastError,
    lastHeartbeat,
    currentModel,
    supportedModes,
    retryCount,
    connect,
    disconnect,
    sendMessage,
    clearCaptions,
    exportCaptions,
    addCaption
  };
}
