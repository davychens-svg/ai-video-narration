import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Language, getTranslations } from '../lib/i18n';

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  settings: {
    serverUrl: string;
    websocketUrl: string;
    videoQuality: 'low' | 'medium' | 'high';
    framerate: number;
    captureInterval: number;
    autoReconnect: boolean;
    maxRetries: number;
    debugMode: boolean;
    responseLength: 'short' | 'medium' | 'long';
  };
  onSettingsChange: (settings: any) => void;
  language: Language;
}

export function SettingsDialog({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
  language
}: SettingsDialogProps) {
  const t = getTranslations(language);

  const handleSettingChange = (key: string, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t.settingsDialogTitle}</DialogTitle>
          <DialogDescription>
            {t.settingsDialogDescription}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Video Settings */}
          <div className="space-y-4">
            
            <div className="space-y-2">
              <Label>{t.videoQuality}</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('videoQuality', 'low')}
                  className={`flex-1 ${settings.videoQuality === 'low' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.videoQualityLow}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('videoQuality', 'medium')}
                  className={`flex-1 ${settings.videoQuality === 'medium' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.videoQualityMedium}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('videoQuality', 'high')}
                  className={`flex-1 ${settings.videoQuality === 'high' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.videoQualityHigh}
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label>{t.responseLengthLabel}</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('responseLength', 'short')}
                  className={`flex-1 ${settings.responseLength === 'short' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.responseLengthShort}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('responseLength', 'medium')}
                  className={`flex-1 ${settings.responseLength === 'medium' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.responseLengthMedium}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSettingChange('responseLength', 'long')}
                  className={`flex-1 ${settings.responseLength === 'long' ? 'bg-gradient-to-br from-purple-500/90 to-pink-500/90 text-white border-purple-400/50 hover:from-purple-600/90 hover:to-pink-600/90 backdrop-blur-sm' : 'hover:bg-accent'}`}
                >
                  {t.responseLengthLong}
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">
                {t.responseLengthHint}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>{t.captureInterval}</Label>
                <span className="text-sm text-muted-foreground">{settings.captureInterval}{t.captureIntervalUnit} ({(1000/settings.captureInterval).toFixed(1)} FPS)</span>
              </div>
              <Slider
                value={[settings.captureInterval]}
                onValueChange={(value) => handleSettingChange('captureInterval', value[0])}
                max={5000}
                min={100}
                step={100}
                className="w-full"
              />
              <div className="text-xs text-muted-foreground">
                {t.captureIntervalHint}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>{t.cameraResolutionLabel}</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.videoQuality === 'low' ? '640x480' : settings.videoQuality === 'medium' ? '1280x720' : '1920x1080'}
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                {t.cameraResolutionHint}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              className="flex-1 hover:bg-accent"
            >
              {t.actionCancel}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              className="flex-1 border border-border hover:bg-accent"
            >
              {t.actionSave}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
