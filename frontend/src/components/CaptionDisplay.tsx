import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { MessageSquare, Download, Trash2, Zap } from 'lucide-react';
import { Language, getTranslations } from '../lib/i18n';

export interface Caption {
  id: string;
  timestamp: string;
  content: string;
  model: string;
  confidence?: number;
  feature?: string;
  latency_ms?: number;
}

interface CaptionDisplayProps {
  captions: Caption[];
  isConnected: boolean;
  onClearCaptions: () => void;
  onExportCaptions: () => void;
  language: Language;
}

export function CaptionDisplay({ 
  captions, 
  isConnected, 
  onClearCaptions, 
  onExportCaptions,
  language
}: CaptionDisplayProps) {
  const t = getTranslations(language);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new captions arrive (contained within ScrollArea)
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [captions]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getModelBadgeColor = (model: string) => {
    if (!model) return 'bg-gray-500/10 text-gray-700 border-gray-200';

    switch (model.toLowerCase()) {
      case 'smolvlm':
        return 'bg-blue-500/10 text-blue-700 border-blue-200';
      case 'moondream':
        return 'bg-purple-500/10 text-purple-700 border-purple-200';
      default:
        return 'bg-gray-500/10 text-gray-700 border-gray-200';
    }
  };

  const getFeatureBadgeColor = (feature?: string) => {
    switch (feature?.toLowerCase()) {
      case 'query':
        return 'bg-green-500/10 text-green-700 border-green-200';
      case 'caption':
        return 'bg-yellow-500/10 text-yellow-700 border-yellow-200';
      case 'detection':
        return 'bg-red-500/10 text-red-700 border-red-200';
      case 'point':
        return 'bg-indigo-500/10 text-indigo-700 border-indigo-200';
      default:
        return 'bg-gray-500/10 text-gray-700 border-gray-200';
    }
  };

  const getPerformanceBadgeColor = (latency_ms?: number) => {
    if (!latency_ms) return 'bg-gray-500/10 text-gray-700 border-gray-200';

    // Fast: < 1s (green), Medium: 1-3s (yellow), Slow: > 3s (red)
    if (latency_ms < 1000) {
      return 'bg-green-500/10 text-green-700 border-green-200';
    } else if (latency_ms < 3000) {
      return 'bg-yellow-500/10 text-yellow-700 border-yellow-200';
    } else {
      return 'bg-red-500/10 text-red-700 border-red-200';
    }
  };

  const formatLatency = (latency_ms?: number) => {
    if (!latency_ms) return null;

    if (latency_ms < 1000) {
      return `${Math.round(latency_ms)}ms`;
    } else {
      return `${(latency_ms / 1000).toFixed(1)}s`;
    }
  };

  const captionCountText = `${captions.length} ${captions.length === 1 ? t.captionSingular : t.captionPlural}`;

  return (
    <Card className="w-full h-[700px] flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            {t.captionPanelTitle}
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onExportCaptions}
              disabled={captions.length === 0}
              className="flex items-center gap-1"
            >
              <Download className="w-3 h-3" />
              {t.export}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onClearCaptions}
              disabled={captions.length === 0}
              className="flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              {t.clear}
            </Button>
          </div>
        </div>
        <div className="text-sm text-muted-foreground">
          {captionCountText} • {isConnected ? t.captionsStatusConnected : t.captionsStatusDisconnected}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        <ScrollArea ref={scrollAreaRef} className="flex-1 px-6 h-full">
          {captions.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <MessageSquare className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="font-medium mb-2">{t.captionEmptyTitle}</h3>
              <p className="text-sm text-muted-foreground max-w-sm">
                {t.captionEmptyMessage}
              </p>
            </div>
          ) : (
            <div className="space-y-4 pb-4 pt-2">
              {captions.map((caption) => (
                <div
                  key={caption.id}
                  className="p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge
                        variant="outline"
                        className={`text-xs ${getModelBadgeColor(caption.model)}`}
                      >
                        {caption.model}
                      </Badge>
                      {caption.feature && (
                        <Badge
                          variant="outline"
                          className={`text-xs ${getFeatureBadgeColor(caption.feature)}`}
                        >
                          {caption.feature}
                        </Badge>
                      )}
                      {caption.confidence && (
                        <Badge variant="outline" className="text-xs">
                          {Math.round(caption.confidence * 100)}{t.captionConfidenceSuffix}
                        </Badge>
                      )}
                      {caption.latency_ms && (
                      <Badge
                        variant="outline"
                        className={`text-xs ${getPerformanceBadgeColor(caption.latency_ms)}`}
                      >
                        <Zap className="w-3 h-3 mr-1" />
                        {formatLatency(caption.latency_ms)}
                      </Badge>
                    )}
                  </div>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {formatTimestamp(caption.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed break-words whitespace-pre-wrap">{caption.content}</p>
                </div>
              ))}
              <div ref={endRef} />
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
