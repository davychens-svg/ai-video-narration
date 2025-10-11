import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Play, Square, Camera, Youtube } from 'lucide-react';
import { VideoOverlay } from './VideoOverlay';
import { Language, getTranslations } from '../lib/i18n';

interface Detection {
  bbox: [number, number, number, number];
  label: string;
  confidence?: number;
}

interface Point {
  x: number;
  y: number;
  label?: string;
}

interface VideoStreamingProps {
  onStreamReady: (stream: MediaStream | null) => void;
  isConnected: boolean;
  captureInterval?: number;
  videoQuality?: 'low' | 'medium' | 'high';
  serverUrl?: string;
  detections?: Detection[];
  points?: Point[];
  overlayMode?: 'detection' | 'point' | 'mask' | 'none';
  backend?: 'transformers' | 'llamacpp';
  prompt?: string;
  responseLength?: 'short' | 'medium' | 'long';
  modelReady?: boolean;
  language: Language;
}

export function VideoStreaming({
  onStreamReady,
  isConnected,
  captureInterval = 500,
  videoQuality = 'medium',
  serverUrl = 'http://localhost:8001',
  detections = [],
  points = [],
  overlayMode = 'none',
  backend = 'llamacpp',
  prompt = 'What objects are visible in this scene?',
  responseLength = 'medium',
  modelReady = true,
  language
}: VideoStreamingProps) {
  const t = useMemo(() => getTranslations(language), [language]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [activeTab, setActiveTab] = useState('camera');
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const startCameraStream = async () => {
    try {
      setError(null);

      // Determine resolution based on video quality setting
      const resolutions = {
        low: { width: 640, height: 480 },
        medium: { width: 1280, height: 720 },
        high: { width: 1920, height: 1080 }
      };
      const resolution = resolutions[videoQuality];

      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
          facingMode: 'user' // Use front camera by default
        },
        audio: false
      });

      // Display in video element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      console.log(`Camera stream started at ${resolution.width}x${resolution.height}`);
      onStreamReady(stream);
      setIsStreaming(true);

      // Start HTTP-based frame capture
      await setupHTTPFrameCapture();
    } catch (err) {
      setError(t.errorCameraAccess);
      console.error('Camera error:', err);
    }
  };

  const setupHTTPFrameCapture = useCallback(async () => {
    try {
      if (!videoRef.current || !canvasRef.current) {
        console.error('Video or canvas ref not available');
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        console.error('Could not get canvas context');
        return;
      }

      console.log('Setting up HTTP frame capture...');

      // Wait for video to be ready
      await new Promise((resolve) => {
        if (video.readyState >= 2) {
          resolve(null);
        } else {
          video.onloadedmetadata = () => resolve(null);
        }
      });

      // Set canvas dimensions
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      let isProcessing = false;

      // Function to capture and send frame
      const captureAndSendFrame = async () => {
        if (isProcessing || !video || video.readyState < 2) {
          return;
        }

        if (!modelReady) {
          return;
        }

        isProcessing = true;

        try {
          // Capture frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // JPEG quality based on video quality setting
          const jpegQuality = videoQuality === 'low' ? 0.6 : videoQuality === 'medium' ? 0.8 : 0.95;
          const imageData = canvas.toDataURL('image/jpeg', jpegQuality);

          // Choose endpoint based on backend
          const endpoint = backend === 'llamacpp'
            ? '/api/process_frame_llamacpp'
            : '/api/process_frame';

          // Send to backend with prompt and response length
          const response = await fetch(`${serverUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image: imageData,
              prompt: prompt,
              custom_query: prompt,
              response_length: responseLength
            })
          });

          if (response.ok) {
            const result = await response.json();
            console.log('Frame processed:', result);

            // Broadcast result via WebSocket format
            const message = {
              type: 'caption',
              timestamp: Date.now() / 1000,
              data: result
            };

            // Trigger custom event that App.tsx can listen to
            window.dispatchEvent(new CustomEvent('frame-result', { detail: message }));
          }
        } catch (error) {
          console.error('Frame capture error:', error);
        } finally {
          isProcessing = false;
        }
      };

      // Capture frames at configured interval
      frameIntervalRef.current = window.setInterval(captureAndSendFrame, captureInterval);
      console.log(`HTTP frame capture started at ${captureInterval}ms interval (${(1000/captureInterval).toFixed(1)} FPS)`);

      // Send first frame immediately
      captureAndSendFrame();
    } catch (err) {
      console.error('Failed to set up frame capture:', err);
    }
  }, [videoQuality, backend, serverUrl, prompt, responseLength, captureInterval, modelReady, language, t]);

  const stopCameraStream = () => {
    // Stop frame capture interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // Stop video tracks
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    onStreamReady(null);
    setIsStreaming(false);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file);
      setYoutubeUrl(''); // Clear URL if file is selected
    } else {
      setError(t.errorInvalidVideo);
    }
  };

  const startYouTubeStream = async () => {
    try {
      setError(null);

      if (!videoRef.current) {
        console.error('Video ref not available');
        setError(t.errorVideoNotReady);
        return;
      }

      let videoSrc = '';

      // Prioritize uploaded file
      if (uploadedFile) {
        videoSrc = URL.createObjectURL(uploadedFile);
        console.log('Using uploaded file:', uploadedFile.name);
      } else if (youtubeUrl.trim()) {
        // Check if it's a YouTube URL or direct video URL
        const videoId = extractYouTubeVideoId(youtubeUrl);

        if (videoId) {
          // It's a YouTube URL - we can't directly access it due to CORS
          setError(t.errorYouTubeCors);
          return;
        } else {
          // Assume it's a direct video URL
          videoSrc = youtubeUrl.trim();
          console.log('Using video URL:', videoSrc);
        }
      } else {
        setError(t.errorVideoRequired);
        return;
      }

      console.log('Loading video from:', videoSrc);

      // Set video source
      videoRef.current.src = videoSrc;
      videoRef.current.load(); // Force load

      // Wait for video to load
      videoRef.current.onloadedmetadata = async () => {
        console.log('Video metadata loaded, dimensions:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
        setIsStreaming(true);

        // Small delay to ensure video is ready
        setTimeout(async () => {
          await setupHTTPFrameCapture();
        }, 500);
      };

      videoRef.current.onloadeddata = () => {
        console.log('Video data loaded');
      };

      videoRef.current.oncanplay = () => {
        console.log('Video can play');
      };

      videoRef.current.onerror = (e) => {
        console.error('Video load error event:', e);
        console.error('Video error details:', videoRef.current?.error);
        setError(t.errorVideoLoad);
        setIsStreaming(false);
      };
    } catch (err) {
      setError(t.errorVideoLoadGeneric);
      console.error('Video load exception:', err);
    }
  };


  const extractYouTubeVideoId = (url: string): string | null => {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
  };

  const stopYouTubeStream = () => {
    // Stop frame capture interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // Revoke object URL if it was created from file
    if (videoRef.current && uploadedFile) {
      URL.revokeObjectURL(videoRef.current.src);
      videoRef.current.src = '';
    }

    onStreamReady(null);
    setIsStreaming(false);
  };

  useEffect(() => {
    // Stop streaming when switching tabs
    if (isStreaming) {
      if (activeTab === 'camera') {
        stopYouTubeStream();
      } else if (activeTab === 'youtube') {
        stopCameraStream();
      }
    }
  }, [activeTab]);

  useEffect(() => {
    // Restart frame capture when captureInterval, responseLength, or backend changes (only if already streaming)
    // Note: prompt changes don't trigger restart - the latest prompt value is always used via closure
    if (isStreaming && frameIntervalRef.current) {
      console.log(`Updating capture settings - backend: ${backend}, interval: ${captureInterval}ms, responseLength: ${responseLength}`);
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
      setupHTTPFrameCapture();
    }
  }, [captureInterval, responseLength, backend, modelReady, isStreaming, setupHTTPFrameCapture]);

  useEffect(() => {
    return () => {
      // Cleanup on component unmount
      stopCameraStream();
      stopYouTubeStream();
    };
  }, []);

  return (
    <div className="w-full p-6">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-xl font-semibold text-foreground/90">{t.videoInputTitle}</h2>
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400 glow-green' : 'bg-red-400'} transition-all duration-300`} />
        </div>
      </div>
      <div>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="camera" className="flex items-center gap-2">
              <Camera className="w-4 h-4" />
              {t.tabCamera}
            </TabsTrigger>
            <TabsTrigger value="youtube" className="flex items-center gap-2">
              <Youtube className="w-4 h-4" />
              {t.tabVideo}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="camera" className="space-y-4">
            <div className="aspect-video glass rounded-2xl overflow-hidden border border-white/10 relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <VideoOverlay
                videoRef={videoRef}
                detections={detections}
                points={points}
                mode={overlayMode}
                language={language}
              />
            </div>

            {/* Hidden canvas for frame capture - camera tab */}
            <canvas ref={activeTab === 'camera' ? canvasRef : null} style={{ display: 'none' }} />

            <div className="flex gap-2">
              {!isStreaming ? (
                <Button onClick={startCameraStream} className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  {t.buttonStartCamera}
                </Button>
              ) : (
                <Button onClick={stopCameraStream} variant="destructive" className="flex items-center gap-2">
                  <Square className="w-4 h-4" />
                  {t.buttonStopCamera}
                </Button>
              )}
            </div>
          </TabsContent>

          <TabsContent value="youtube" className="space-y-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="youtube-url" className="text-foreground">{t.videoUrlLabel}</Label>
                <Input
                  id="youtube-url"
                  type="text"
                  placeholder={t.videoUrlPlaceholder}
                  value={youtubeUrl}
                  onChange={(e) => {
                    setYoutubeUrl(e.target.value);
                    setUploadedFile(null);
                    setError(null);
                  }}
                  className="!bg-black/30 !text-white border-white/20 placeholder:text-white/50"
                />
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setYoutubeUrl('https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4');
                      setUploadedFile(null);
                      setError(null);
                    }}
                    className="text-xs text-blue-300 hover:text-blue-200"
                  >
                    {t.sampleVideo1}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setYoutubeUrl('https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4');
                      setUploadedFile(null);
                      setError(null);
                    }}
                    className="text-xs text-blue-300 hover:text-blue-200"
                  >
                    {t.sampleVideo2}
                  </Button>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex-1 border-t border-white/10"></div>
                <span className="text-xs text-foreground/60">{t.orDividerText}</span>
                <div className="flex-1 border-t border-white/10"></div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="video-file" className="text-foreground">{t.uploadVideoLabel}</Label>
                <input
                  ref={fileInputRef}
                  id="video-file"
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full border-white/20 hover:bg-white/10"
                >
                  {uploadedFile ? uploadedFile.name : t.chooseVideoFile}
                </Button>
                {uploadedFile && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setUploadedFile(null);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                    className="w-full text-red-300 hover:text-red-200"
                  >
                    {t.clearFile}
                  </Button>
                )}
              </div>

              <div className="p-3 glass rounded-xl border border-blue-300/20">
                <p className="text-xs text-foreground/80 mb-2">
                  <strong>{t.tipsTitle}</strong>
                </p>
                <ul className="text-xs text-foreground/70 space-y-1 list-disc list-inside">
                  {t.tipsList.map((tip) => (
                    <li key={tip}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="aspect-video glass rounded-2xl overflow-hidden border border-white/10 relative">
              {activeTab === 'youtube' && (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    loop
                    controls
                    crossOrigin="anonymous"
                    className="w-full h-full object-cover"
                  />
                  <VideoOverlay
                    videoRef={videoRef}
                    detections={detections}
                    points={points}
                    mode={overlayMode}
                    language={language}
                  />
                </>
              )}
             {!isStreaming && (
                <div className="absolute inset-0 flex items-center justify-center text-foreground/60 bg-black/20">
                  {t.videoPlaceholder}
                </div>
              )}
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            <div className="flex gap-2">
              {!isStreaming ? (
                <Button onClick={startYouTubeStream} className="flex items-center gap-2" disabled={!youtubeUrl && !uploadedFile}>
                  <Play className="w-4 h-4" />
                  {t.buttonLoadVideo}
                </Button>
              ) : (
                <Button onClick={stopYouTubeStream} variant="destructive" className="flex items-center gap-2">
                  <Square className="w-4 h-4" />
                  {t.buttonStopVideo}
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
