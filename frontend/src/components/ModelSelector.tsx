import { Label } from './ui/label';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Brain, Eye, Search, Target, Send, Check } from 'lucide-react';
import { Language, getTranslations, FeatureKey } from '../lib/i18n';

export type ModelType = 'smolvlm' | 'moondream';
export type MoondreamFeature = 'query' | 'caption' | 'detection' | 'point' | 'mask';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
  moondreamFeature: MoondreamFeature;
  onMoondreamFeatureChange: (feature: MoondreamFeature) => void;
  customQuery: string;
  onCustomQueryChange: (query: string) => void;
  onSendQuery?: () => void;
  onMoondreamSubmit?: () => void;
  isProcessing: boolean;
  responseLength: 'short' | 'medium' | 'long';
  language: Language;
}

export function ModelSelector({
  selectedModel,
  onModelChange,
  moondreamFeature,
  onMoondreamFeatureChange,
  customQuery,
  onCustomQueryChange,
  onSendQuery,
  onMoondreamSubmit,
  isProcessing,
  responseLength,
  language
}: ModelSelectorProps) {
  const t = getTranslations(language);
  const modelInfo = {
    smolvlm: {
      name: t.modelSmolVLM,
      description: t.modelSmolVLMDesc,
      icon: <Brain className="w-4 h-4" />,
      features: t.modelSmolFeatures
    },
    moondream: {
      name: t.modelMoondream,
      description: t.modelMoondreamDesc,
      icon: <Eye className="w-4 h-4" />,
      features: t.modelMoondreamFeatures
    }
  };

  const moondreamFeatureInfo: Record<FeatureKey, { name: string; description: string; icon: JSX.Element }> = {
    query: {
      name: t.moondreamFeatureNames.query,
      description: t.moondreamFeatureDescriptions.query,
      icon: <Search className="w-4 h-4" />
    },
    caption: {
      name: t.moondreamFeatureNames.caption,
      description: t.moondreamFeatureDescriptions.caption,
      icon: <Eye className="w-4 h-4" />
    },
    detection: {
      name: t.moondreamFeatureNames.detection,
      description: t.moondreamFeatureDescriptions.detection,
      icon: <Target className="w-4 h-4" />
    },
    point: {
      name: t.moondreamFeatureNames.point,
      description: t.moondreamFeatureDescriptions.point,
      icon: <Target className="w-4 h-4" />
    },
    mask: {
      name: t.moondreamFeatureNames.mask,
      description: t.moondreamFeatureDescriptions.mask,
      icon: <Eye className="w-4 h-4" />
    }
  };

  const responseLengthLabels: Record<'short' | 'medium' | 'long', string> = {
    short: t.responseLengthShort,
    medium: t.responseLengthMedium,
    long: t.responseLengthLong
  };

  return (
    <div className="w-full p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-foreground/90">{t.modelSectionTitle}</h2>
      </div>
      <div className="space-y-6">
        {/* Model Selection */}
        <div className="space-y-2">
          <Label>{t.modelSelectLabel}</Label>
          <div className="grid grid-cols-2 gap-3">
            <Button
              variant="outline"
              className={`h-auto py-4 px-4 ${
                selectedModel === 'smolvlm'
                  ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm'
                  : 'hover:bg-accent'
              }`}
              onClick={() => onModelChange('smolvlm')}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-2 w-full">
                {modelInfo.smolvlm.icon}
                <div className="flex flex-col items-start flex-1">
                  <span className="font-semibold text-sm">{modelInfo.smolvlm.name}</span>
                  <span className="text-xs opacity-80">{t.modelSmolTagline}</span>
                </div>
                {selectedModel === 'smolvlm' && <Check className="w-5 h-5" />}
              </div>
            </Button>
            <Button
              variant="outline"
              className={`h-auto py-4 px-4 ${
                selectedModel === 'moondream'
                  ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm'
                  : 'hover:bg-accent'
              }`}
              onClick={() => onModelChange('moondream')}
              disabled={isProcessing}
            >
              <div className="flex items-center gap-2 w-full">
                {modelInfo.moondream.icon}
                <div className="flex flex-col items-start flex-1">
                  <span className="font-semibold text-sm">{modelInfo.moondream.name}</span>
                  <span className="text-xs opacity-80">{t.modelMoondreamTagline}</span>
                </div>
                {selectedModel === 'moondream' && <Check className="w-5 h-5" />}
              </div>
            </Button>
          </div>
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

        {/* SmolVLM Query */}
        {selectedModel === 'smolvlm' && (
          <div className="space-y-4">
            {/* Real-time Status */}
            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
              <p className="text-xs text-green-400 flex items-center gap-2">
                <Check className="w-4 h-4" />
                <span className="font-medium">{t.smolRealtimeTitle}</span>
                <span className="opacity-70">{t.smolRealtimeSubtitle}</span>
              </p>
            </div>

            {/* Prompt */}
            <div className="space-y-2">
              <Label htmlFor="smolvlm-prompt">
                {t.smolPromptLabel} <span className="text-red-500">*</span>
              </Label>
              <Textarea
                id="smolvlm-prompt"
                placeholder={t.smolPromptPlaceholder}
                value={customQuery}
                onChange={(e) => onCustomQueryChange(e.target.value)}
                onFocus={(e) => e.target.select()}
                disabled={isProcessing}
                className="min-h-[80px] bg-background/50 border-input text-foreground placeholder:text-muted-foreground"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && customQuery.trim() && onSendQuery) {
                    e.preventDefault();
                    onSendQuery();
                  }
                }}
              />
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-green-400 flex items-center gap-1 flex-1">
                  <Check className="w-3 h-3" />
                  {t.smolPromptHint}
                </p>
                <Button
                  size="sm"
                  onClick={onSendQuery}
                  disabled={!customQuery.trim() || isProcessing}
                  className="flex items-center gap-2 shrink-0"
                >
                  <Send className="w-3 h-3" />
                  {t.actionUpdate}
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Moondream Features */}
        {selectedModel === 'moondream' && (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>{t.moondreamFeatureLabel}</Label>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(moondreamFeatureInfo).map(([key, info]) => (
                  <Button
                    key={key}
                    variant="outline"
                    className={`h-auto py-3 px-3 ${
                      moondreamFeature === key
                        ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm'
                        : 'hover:bg-accent'
                    }`}
                    onClick={() => onMoondreamFeatureChange(key as MoondreamFeature)}
                    disabled={isProcessing}
                  >
                    <div className="flex items-center gap-2 w-full">
                      {info.icon}
                      <span className="text-sm font-medium flex-1 text-left">{info.name}</span>
                      {moondreamFeature === key && <Check className="w-4 h-4" />}
                    </div>
                  </Button>
                ))}
              </div>
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

            {moondreamFeature === 'caption' && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{t.responseLengthPreference}</span>
                <Badge variant="outline" className="uppercase tracking-wide">
                  {responseLengthLabels[responseLength]}
                </Badge>
              </div>
            )}

            {/* Custom Query Input */}
            {moondreamFeature === 'query' && (
              <div className="space-y-2">
                <Label htmlFor="custom-query">{t.customQuery}</Label>
                <Textarea
                  id="custom-query"
                  placeholder={t.customQueryPlaceholder}
                  value={customQuery}
                  onChange={(e) => onCustomQueryChange(e.target.value)}
                  onFocus={(e) => e.target.select()}
                  disabled={isProcessing}
                  className="min-h-[80px] bg-background/50 border-input text-foreground placeholder:text-muted-foreground"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && customQuery.trim() && onMoondreamSubmit) {
                      e.preventDefault();
                      onMoondreamSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-green-400 flex items-center gap-1 flex-1">
                    <Check className="w-3 h-3" />
                    {t.queryInstruction}
                  </p>
                  <Button
                    size="sm"
                    onClick={onMoondreamSubmit}
                    disabled={!customQuery.trim() || isProcessing}
                    className="flex items-center gap-2 shrink-0"
                  >
                    <Send className="w-3 h-3" />
                    {t.actionUpdate}
                  </Button>
                </div>
              </div>
            )}

            {/* Object Detection Input */}
            {moondreamFeature === 'detection' && (
              <div className="space-y-2">
                <Label htmlFor="detect-object">{t.moondreamFeatureNames.detection}</Label>
                <Input
                  id="detect-object"
                  placeholder={t.detectionPlaceholder}
                  value={customQuery}
                  onChange={(e) => onCustomQueryChange(e.target.value)}
                  onFocus={(e) => e.target.select()}
                  disabled={isProcessing}
                  className="bg-background/50 border-input text-foreground placeholder:text-muted-foreground"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && onMoondreamSubmit) {
                      e.preventDefault();
                      onMoondreamSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-green-400 flex items-center gap-1 flex-1">
                    <Check className="w-3 h-3" />
                    {t.detectionInstruction}
                  </p>
                  <Button
                    size="sm"
                    onClick={onMoondreamSubmit}
                    disabled={isProcessing}
                    className="flex items-center gap-2 shrink-0"
                  >
                    <Send className="w-3 h-3" />
                    {t.actionUpdate}
                  </Button>
                </div>
              </div>
            )}

            {/* Point Detection Input */}
            {moondreamFeature === 'point' && (
              <div className="space-y-2">
                <Label htmlFor="point-object">{t.moondreamFeatureNames.point}</Label>
                <Input
                  id="point-object"
                  placeholder={t.pointInstruction}
                  value={customQuery}
                  onChange={(e) => onCustomQueryChange(e.target.value)}
                  onFocus={(e) => e.target.select()}
                  disabled={isProcessing}
                  className="bg-background/50 border-input text-foreground placeholder:text-muted-foreground"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customQuery.trim() && onMoondreamSubmit) {
                      e.preventDefault();
                      onMoondreamSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-green-400 flex items-center gap-1 flex-1">
                    <Check className="w-3 h-3" />
                    {t.pointInstruction}
                  </p>
                  <Button
                    size="sm"
                    onClick={onMoondreamSubmit}
                    disabled={!customQuery.trim() || isProcessing}
                    className="flex items-center gap-2 shrink-0"
                  >
                    <Send className="w-3 h-3" />
                    {t.actionUpdate}
                  </Button>
                </div>
              </div>
            )}

            {/* Privacy Mask Input */}
            {moondreamFeature === 'mask' && (
              <div className="space-y-2">
                <Label htmlFor="mask-object">{t.moondreamFeatureNames.mask}</Label>
                <Input
                  id="mask-object"
                  placeholder={t.maskInstruction}
                  value={customQuery}
                  onChange={(e) => onCustomQueryChange(e.target.value)}
                  onFocus={(e) => e.target.select()}
                  disabled={isProcessing}
                  className="bg-background/50 border-input text-foreground placeholder:text-muted-foreground"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customQuery.trim() && onMoondreamSubmit) {
                      e.preventDefault();
                      onMoondreamSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs text-green-400 flex items-center gap-1 flex-1">
                    <Check className="w-3 h-3" />
                    {t.maskInstruction}
                  </p>
                  <Button
                    size="sm"
                    onClick={onMoondreamSubmit}
                    disabled={!customQuery.trim() || isProcessing}
                    className="flex items-center gap-2 shrink-0"
                  >
                    <Send className="w-3 h-3" />
                    {t.actionUpdate}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <div className="flex items-center gap-2 p-3 bg-primary/10 rounded-lg">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">{t.processingFrames}</span>
          </div>
        )}
      </div>
    </div>
  );
}
