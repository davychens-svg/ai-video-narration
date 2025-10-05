import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { MessageSquare, Download, Trash2 } from 'lucide-react';

export interface Caption {
  id: string;
  timestamp: string;
  content: string;
  model: string;
  confidence?: number;
  feature?: string;
}

interface CaptionDisplayProps {
  captions: Caption[];
  isConnected: boolean;
  onClearCaptions: () => void;
  onExportCaptions: () => void;
}

export function CaptionDisplay({ 
  captions, 
  isConnected, 
  onClearCaptions, 
  onExportCaptions 
}: CaptionDisplayProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new captions arrive
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [captions]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getModelBadgeColor = (model: string) => {
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

  return (
    <Card className="w-full h-[500px] flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Real-time Captions
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
              Export
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onClearCaptions}
              disabled={captions.length === 0}
              className="flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </Button>
          </div>
        </div>
        <div className="text-sm text-muted-foreground">
          {captions.length} caption{captions.length !== 1 ? 's' : ''} • 
          {isConnected ? ' Connected to WebSocket' : ' Disconnected'}
        </div>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col p-0">
        <ScrollArea className="flex-1 px-6" ref={scrollAreaRef}>
          {captions.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-8">
              <MessageSquare className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="font-medium mb-2">No captions yet</h3>
              <p className="text-sm text-muted-foreground max-w-sm">
                Start video streaming and ensure the WebSocket connection is active to receive real-time captions.
              </p>
            </div>
          ) : (
            <div className="space-y-4 pb-4">
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
                          {Math.round(caption.confidence * 100)}% confident
                        </Badge>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {formatTimestamp(caption.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed">{caption.content}</p>
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