import { Button } from './ui/button';
import { 
  Wifi, 
  WifiOff, 
  Server, 
  Video, 
  Settings
} from 'lucide-react';

interface ConnectionStatusProps {
  webSocketConnected: boolean;
  backendConnected: boolean;
  videoStreamActive: boolean;
  onOpenSettings: () => void;
  serverUrl: string;
}

export function ConnectionStatus({
  webSocketConnected,
  backendConnected,
  videoStreamActive,
  onOpenSettings,
  serverUrl
}: ConnectionStatusProps) {
  const getStatusColor = (connected: boolean) => {
    return connected ? 'bg-green-500' : 'bg-red-500';
  };

  const getStatusText = (connected: boolean) => {
    return connected ? 'Connected' : 'Disconnected';
  };

  return (
    <div className="w-full p-6">
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-foreground/90">System Status</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenSettings}
            className="flex items-center gap-1"
          >
            <Settings className="w-3 h-3" />
            Settings
          </Button>
        </div>

        <div className="space-y-3">
          {/* WebSocket Connection */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {webSocketConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
              <span className="text-sm">WebSocket</span>
            </div>
            <div className={`glass px-3 py-1.5 rounded-full ${webSocketConnected ? 'glow-green' : ''}`}>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${getStatusColor(webSocketConnected)}`} />
                <span className="text-sm text-foreground/80">{getStatusText(webSocketConnected)}</span>
              </div>
            </div>
          </div>

          {/* Backend Connection */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              <span className="text-sm">Backend Server</span>
            </div>
            <div className={`glass px-3 py-1.5 rounded-full ${backendConnected ? 'glow-cyan' : ''}`}>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${getStatusColor(backendConnected)}`} />
                <span className="text-sm text-foreground/80">{getStatusText(backendConnected)}</span>
              </div>
            </div>
          </div>

          {/* Video Stream */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Video className="w-4 h-4" />
              <span className="text-sm">Video Stream</span>
            </div>
            <div className={`glass px-3 py-1.5 rounded-full ${videoStreamActive ? 'glow-teal' : ''}`}>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${getStatusColor(videoStreamActive)}`} />
                <span className="text-sm text-foreground/80">{videoStreamActive ? 'Active' : 'Inactive'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Connection Details */}
        <div className="mt-4 pt-4 glass p-4 rounded-2xl">
          <div className="space-y-2 text-xs text-foreground/60">
            <div className="flex justify-between">
              <span>Server URL:</span>
              <span className="font-mono text-blue-300">{serverUrl}</span>
            </div>
            <div className="flex justify-between">
              <span>Protocol:</span>
              <span className="text-cyan-300">WebRTC + WebSocket</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
