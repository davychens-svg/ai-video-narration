import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

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
  };
  onSettingsChange: (settings: any) => void;
}

export function SettingsDialog({ 
  open, 
  onOpenChange, 
  settings, 
  onSettingsChange 
}: SettingsDialogProps) {
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
          <DialogTitle>Video Processing Settings</DialogTitle>
          <DialogDescription>
            Configure video capture quality and frame processing rate.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Video Settings */}
          <div className="space-y-4">
            
            <div className="space-y-2">
              <Label>Video Quality</Label>
              <Select 
                value={settings.videoQuality} 
                onValueChange={(value) => handleSettingChange('videoQuality', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low (480p)</SelectItem>
                  <SelectItem value="medium">Medium (720p)</SelectItem>
                  <SelectItem value="high">High (1080p)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Capture Interval</Label>
                <span className="text-sm text-muted-foreground">{settings.captureInterval}ms ({(1000/settings.captureInterval).toFixed(1)} FPS)</span>
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
                How often to capture and process frames. Lower = faster updates but higher processing load.
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Camera Resolution</Label>
                <span className="text-sm text-muted-foreground">
                  {settings.videoQuality === 'low' ? '640x480' : settings.videoQuality === 'medium' ? '1280x720' : '1920x1080'}
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                Higher resolution provides more detail but may slow down processing.
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              onClick={() => onOpenChange(false)}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button 
              onClick={() => onOpenChange(false)}
              className="flex-1"
            >
              Save Settings
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}