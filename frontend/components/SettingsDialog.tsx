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
          <DialogTitle>Connection Settings</DialogTitle>
          <DialogDescription>
            Configure your backend server connection and video processing settings.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Server Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium">Server Configuration</h4>
            
            <div className="space-y-2">
              <Label htmlFor="server-url">Backend Server URL</Label>
              <Input
                id="server-url"
                placeholder="http://localhost:8000"
                value={settings.serverUrl}
                onChange={(e) => handleSettingChange('serverUrl', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="websocket-url">WebSocket URL</Label>
              <Input
                id="websocket-url"
                placeholder="ws://localhost:8000/ws"
                value={settings.websocketUrl}
                onChange={(e) => handleSettingChange('websocketUrl', e.target.value)}
              />
            </div>
          </div>

          <Separator />

          {/* Video Settings */}
          <div className="space-y-4">
            <h4 className="font-medium">Video Processing</h4>
            
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
                <Label>Frame Rate</Label>
                <span className="text-sm text-muted-foreground">{settings.framerate} FPS</span>
              </div>
              <Slider
                value={[settings.framerate]}
                onValueChange={(value) => handleSettingChange('framerate', value[0])}
                max={30}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
          </div>

          <Separator />

          {/* Connection Settings */}
          <div className="space-y-4">
            <h4 className="font-medium">Connection Options</h4>
            
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto Reconnect</Label>
                <div className="text-xs text-muted-foreground">
                  Automatically reconnect when connection is lost
                </div>
              </div>
              <Switch
                checked={settings.autoReconnect}
                onCheckedChange={(checked) => handleSettingChange('autoReconnect', checked)}
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Max Retry Attempts</Label>
                <span className="text-sm text-muted-foreground">{settings.maxRetries}</span>
              </div>
              <Slider
                value={[settings.maxRetries]}
                onValueChange={(value) => handleSettingChange('maxRetries', value[0])}
                max={10}
                min={1}
                step={1}
                className="w-full"
                disabled={!settings.autoReconnect}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Debug Mode</Label>
                <div className="text-xs text-muted-foreground">
                  Show detailed connection logs in console
                </div>
              </div>
              <Switch
                checked={settings.debugMode}
                onCheckedChange={(checked) => handleSettingChange('debugMode', checked)}
              />
            </div>
          </div>

          <Separator />

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