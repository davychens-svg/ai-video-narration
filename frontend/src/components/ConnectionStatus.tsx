import { Button } from './ui/button';
import {
  Wifi,
  WifiOff,
  Server,
  Video,
  Settings
} from 'lucide-react';
import { Language, getTranslations } from '../lib/i18n';

interface ConnectionStatusProps {
  webSocketConnected: boolean;
  backendConnected: boolean;
  videoStreamActive: boolean;
  onOpenSettings: () => void;
  serverUrl: string;
  language: Language;
}

export function ConnectionStatus({
  webSocketConnected,
  backendConnected,
  videoStreamActive,
  onOpenSettings,
  serverUrl,
  language
}: ConnectionStatusProps) {
  const t = getTranslations(language);
  const getStatusColor = (connected: boolean) => {
    return connected ? 'bg-green-500' : 'bg-red-500';
  };

  const getStatusText = (connected: boolean) => {
    return connected ? t.connected : t.disconnected;
  };

  return (
    <div className="w-full p-6">
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-foreground/90">{t.connectionSystemStatus}</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenSettings}
            className="flex items-center gap-1"
          >
            <Settings className="w-3 h-3" />
            {t.connectionSettingsButton}
          </Button>
        </div>

        <div className="space-y-3">
          {/* WebSocket Connection */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {webSocketConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
              <span className="text-sm">{t.connectionWebsocket}</span>
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
              <span className="text-sm">{t.connectionBackend}</span>
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
              <span className="text-sm">{t.connectionVideoStream}</span>
            </div>
            <div className={`glass px-3 py-1.5 rounded-full ${videoStreamActive ? 'glow-teal' : ''}`}>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${getStatusColor(videoStreamActive)}`} />
                <span className="text-sm text-foreground/80">{videoStreamActive ? t.statusActive : t.statusInactive}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Connection Details */}
        <div className="mt-4 pt-4 glass p-4 rounded-2xl">
          <div className="space-y-2 text-xs text-foreground/60">
            <div className="flex justify-between">
              <span>{t.serverUrlLabel}</span>
              <span className="font-mono text-blue-300">{serverUrl}</span>
            </div>
            <div className="flex justify-between">
              <span>{t.protocolLabel}</span>
              <span className="text-cyan-300">{t.protocolValue}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
