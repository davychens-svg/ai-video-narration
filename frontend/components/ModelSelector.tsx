import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Brain, Eye, Search, Target } from 'lucide-react';

export type ModelType = 'smolvlm' | 'moondream';
export type MoondreamFeature = 'query' | 'caption' | 'detection' | 'point';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
  moondreamFeature: MoondreamFeature;
  onMoondreamFeatureChange: (feature: MoondreamFeature) => void;
  customQuery: string;
  onCustomQueryChange: (query: string) => void;
  isProcessing: boolean;
}

export function ModelSelector({
  selectedModel,
  onModelChange,
  moondreamFeature,
  onMoondreamFeatureChange,
  customQuery,
  onCustomQueryChange,
  isProcessing
}: ModelSelectorProps) {
  const modelInfo = {
    smolvlm: {
      name: 'SmolVLM',
      description: 'Lightweight vision-language model optimized for speed and efficiency',
      icon: <Brain className="w-4 h-4" />,
      features: ['Fast inference', 'Low memory usage', 'General purpose captions']
    },
    moondream: {
      name: 'Moondream',
      description: 'Advanced vision model with multiple specialized capabilities',
      icon: <Eye className="w-4 h-4" />,
      features: ['Custom queries', 'Object detection', 'Detailed captions', 'Point detection']
    }
  };

  const moondreamFeatureInfo = {
    query: {
      name: 'Custom Query',
      description: 'Ask specific questions about the video content',
      icon: <Search className="w-4 h-4" />
    },
    caption: {
      name: 'Auto Caption',
      description: 'Generate descriptive captions automatically',
      icon: <Eye className="w-4 h-4" />
    },
    detection: {
      name: 'Object Detection',
      description: 'Identify and locate objects in the video',
      icon: <Target className="w-4 h-4" />
    },
    point: {
      name: 'Point Detection',
      description: 'Detect specific points and coordinates',
      icon: <Target className="w-4 h-4" />
    }
  };

  return (
    <div className="w-full p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-foreground/90">AI Model Configuration</h2>
      </div>
      <div className="space-y-6">
        {/* Model Selection */}
        <div className="space-y-2">
          <Label>Select Model</Label>
          <Select 
            value={selectedModel} 
            onValueChange={(value: ModelType) => onModelChange(value)}
            disabled={isProcessing}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="smolvlm">
                <div className="flex items-center gap-2">
                  {modelInfo.smolvlm.icon}
                  SmolVLM
                </div>
              </SelectItem>
              <SelectItem value="moondream">
                <div className="flex items-center gap-2">
                  {modelInfo.moondream.icon}
                  Moondream
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Model Information */}
        <div className="p-4 bg-muted rounded-lg space-y-3">
          <div className="flex items-center gap-2">
            {modelInfo[selectedModel].icon}
            <h3 className="font-medium">{modelInfo[selectedModel].name}</h3>
          </div>
          <p className="text-sm text-muted-foreground">
            {modelInfo[selectedModel].description}
          </p>
          <div className="flex flex-wrap gap-1">
            {modelInfo[selectedModel].features.map((feature, index) => (
              <Badge key={index} variant="secondary" className="text-xs">
                {feature}
              </Badge>
            ))}
          </div>
        </div>

        {/* Moondream Features */}
        {selectedModel === 'moondream' && (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Moondream Feature</Label>
              <Select
                value={moondreamFeature}
                onValueChange={(value: MoondreamFeature) => onMoondreamFeatureChange(value)}
                disabled={isProcessing}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(moondreamFeatureInfo).map(([key, info]) => (
                    <SelectItem key={key} value={key}>
                      <div className="flex items-center gap-2">
                        {info.icon}
                        {info.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Feature Information */}
            <div className="p-3 bg-accent rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                {moondreamFeatureInfo[moondreamFeature].icon}
                <span className="font-medium text-sm">
                  {moondreamFeatureInfo[moondreamFeature].name}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                {moondreamFeatureInfo[moondreamFeature].description}
              </p>
            </div>

            {/* Custom Query Input */}
            {moondreamFeature === 'query' && (
              <div className="space-y-2">
                <Label htmlFor="custom-query">Custom Query</Label>
                <Textarea
                  id="custom-query"
                  placeholder="What would you like to know about the video? e.g., 'What objects are visible in the scene?'"
                  value={customQuery}
                  onChange={(e) => onCustomQueryChange(e.target.value)}
                  disabled={isProcessing}
                  className="min-h-[80px]"
                />
              </div>
            )}
          </div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <div className="flex items-center gap-2 p-3 bg-primary/10 rounded-lg">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">Processing video frames...</span>
          </div>
        )}

        {/* Hardware Info */}
        <div className="glass p-4 rounded-2xl">
          <div className="text-xs text-foreground/60 space-y-2">
            <div className="flex justify-between">
              <span>Platform Support:</span>
              <span className="text-blue-300">macOS, Linux/Ubuntu</span>
            </div>
            <div className="flex justify-between">
              <span>Hardware:</span>
              <span className="text-teal-300">Mac M-Chip, NVIDIA GPU</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}