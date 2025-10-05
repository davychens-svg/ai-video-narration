import React, { useRef, useEffect, useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Play, Square, Camera, Youtube } from 'lucide-react';

interface VideoStreamingProps {
  onStreamReady: (stream: MediaStream | null) => void;
  isConnected: boolean;
}

export function VideoStreaming({ onStreamReady, isConnected }: VideoStreamingProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
  const [activeTab, setActiveTab] = useState('camera');
  const [error, setError] = useState<string | null>(null);

  const startCameraStream = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'environment' // Use back camera by default
        },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      onStreamReady(stream);
      setIsStreaming(true);
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera access error:', err);
    }
  };

  const stopCameraStream = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    onStreamReady(null);
    setIsStreaming(false);
  };

  const startYouTubeStream = () => {
    try {
      setError(null);
      // Extract video ID from YouTube URL
      const videoId = extractYouTubeVideoId(youtubeUrl);
      if (!videoId) {
        setError('Invalid YouTube URL. Please provide a valid YouTube video URL.');
        return;
      }

      if (videoRef.current) {
        // For YouTube videos, we'll embed them directly
        // In a real implementation, you'd need to handle YouTube API integration
        videoRef.current.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&mute=1`;
      }
      
      setIsStreaming(true);
      // Note: For actual YouTube processing, you'd need backend integration
      onStreamReady(null); // YouTube stream would be handled differently
    } catch (err) {
      setError('Failed to load YouTube video.');
      console.error('YouTube load error:', err);
    }
  };

  const extractYouTubeVideoId = (url: string): string | null => {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
  };

  const stopYouTubeStream = () => {
    if (videoRef.current) {
      videoRef.current.src = '';
    }
    onStreamReady(null);
    setIsStreaming(false);
  };

  useEffect(() => {
    return () => {
      // Cleanup on component unmount
      if (activeTab === 'camera') {
        stopCameraStream();
      }
    };
  }, []);

  return (
    <div className="w-full p-6">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-xl font-semibold text-foreground/90">Video Input</h2>
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400 glow-green' : 'bg-red-400'} transition-all duration-300`} />
        </div>
      </div>
      <div>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="camera" className="flex items-center gap-2">
              <Camera className="w-4 h-4" />
              Live Camera
            </TabsTrigger>
            <TabsTrigger value="youtube" className="flex items-center gap-2">
              <Youtube className="w-4 h-4" />
              YouTube Video
            </TabsTrigger>
          </TabsList>

          <TabsContent value="camera" className="space-y-4">
            <div className="aspect-video glass rounded-2xl overflow-hidden border border-white/10">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            </div>
            
            <div className="flex gap-2">
              {!isStreaming ? (
                <Button onClick={startCameraStream} className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Start Camera
                </Button>
              ) : (
                <Button onClick={stopCameraStream} variant="destructive" className="flex items-center gap-2">
                  <Square className="w-4 h-4" />
                  Stop Camera
                </Button>
              )}
            </div>
          </TabsContent>

          <TabsContent value="youtube" className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="youtube-url">YouTube URL</Label>
              <Input
                id="youtube-url"
                type="url"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
              />
            </div>

            <div className="aspect-video glass rounded-2xl overflow-hidden border border-white/10">
              {activeTab === 'youtube' && isStreaming ? (
                <iframe
                  ref={videoRef as any}
                  className="w-full h-full"
                  src={`https://www.youtube.com/embed/${extractYouTubeVideoId(youtubeUrl)}?autoplay=1&mute=1`}
                  allowFullScreen
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-foreground/60">
                  YouTube video will appear here
                </div>
              )}
            </div>

            <div className="flex gap-2">
              {!isStreaming ? (
                <Button onClick={startYouTubeStream} className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Load YouTube Video
                </Button>
              ) : (
                <Button onClick={stopYouTubeStream} variant="destructive" className="flex items-center gap-2">
                  <Square className="w-4 h-4" />
                  Stop Video
                </Button>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {error && (
          <div className="mt-4 p-4 glass rounded-2xl border border-red-300/30">
            <p className="text-sm text-red-200">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}