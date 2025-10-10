import { useRef, useEffect, type RefObject } from 'react';

interface Detection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  label: string;
  confidence?: number;
}

interface Point {
  x: number;
  y: number;
  label?: string;
}

interface VideoOverlayProps {
  videoRef: RefObject<HTMLVideoElement>;
  detections?: Detection[];
  points?: Point[];
  mode?: 'detection' | 'point' | 'none';
}

export function VideoOverlay({ videoRef, detections = [], points = [], mode = 'none' }: VideoOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Match canvas size to video size
    const updateCanvasSize = () => {
      canvas.width = video.videoWidth || video.clientWidth;
      canvas.height = video.videoHeight || video.clientHeight;
    };

    updateCanvasSize();

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const detectionColor = '#3B82F6'; // Blue color for detection boxes
    const pointColor = '#3B82F6'; // Blue color for points

    if (mode === 'detection' && detections.length > 0) {
      // Draw bounding boxes
      detections.forEach((detection) => {
        if (!detection?.bbox || detection.bbox.length < 4) {
          return;
        }

        const [rawX1, rawY1, rawX2, rawY2] = detection.bbox;
        const x1 = Number(rawX1) || 0;
        const y1 = Number(rawY1) || 0;
        const x2 = Number(rawX2) || 0;
        const y2 = Number(rawY2) || 0;

        const width = x2 - x1;
        const height = y2 - y1;

        // Draw box
        ctx.strokeStyle = detectionColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        ctx.fillStyle = detectionColor;
        const labelText = detection.confidence
          ? `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`
          : detection.label;
        const textMetrics = ctx.measureText(labelText);
        ctx.fillRect(x1, y1 - 25, textMetrics.width + 10, 25);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText(labelText, x1 + 5, y1 - 7);
      });
    }

    if (mode === 'point' && points.length > 0) {
      // Draw points
      points.forEach((point) => {
        if (point == null) {
          return;
        }

        const x = Number(point.x) || 0;
        const y = Number(point.y) || 0;
        const color = pointColor;

        // Draw point circle
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw label if available
        if (point.label) {
          ctx.fillStyle = color;
          const textMetrics = ctx.measureText(point.label);
          ctx.fillRect(x + 12, y - 12, textMetrics.width + 10, 24);

          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.fillText(point.label, x + 17, y + 4);
        }
      });
    }

    // Re-draw on video updates
    const interval = setInterval(() => {
      if (mode !== 'none' && (detections.length > 0 || points.length > 0)) {
        updateCanvasSize();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const detectionColor = '#3B82F6'; // Blue color for detection boxes
        const pointColor = '#3B82F6'; // Blue color for points

        // Redraw everything
        if (mode === 'detection' && detections.length > 0) {
          detections.forEach((detection) => {
            if (!detection?.bbox || detection.bbox.length < 4) {
              return;
            }

            const [rawX1, rawY1, rawX2, rawY2] = detection.bbox;
            const x1 = Number(rawX1) || 0;
            const y1 = Number(rawY1) || 0;
            const x2 = Number(rawX2) || 0;
            const y2 = Number(rawY2) || 0;
            const width = x2 - x1;
            const height = y2 - y1;

            ctx.strokeStyle = detectionColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);

            ctx.fillStyle = detectionColor;
            const labelText = detection.confidence
              ? `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`
              : detection.label;
            const textMetrics = ctx.measureText(labelText);
            ctx.fillRect(x1, y1 - 25, textMetrics.width + 10, 25);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px Inter, sans-serif';
            ctx.fillText(labelText, x1 + 5, y1 - 7);
          });
        }

        if (mode === 'point' && points.length > 0) {
          points.forEach((point) => {
            if (point == null) {
              return;
            }

            const x = Number(point.x) || 0;
            const y = Number(point.y) || 0;
            const color = pointColor;

            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();

            if (point.label) {
              ctx.fillStyle = color;
              const textMetrics = ctx.measureText(point.label);
              ctx.fillRect(x + 12, y - 12, textMetrics.width + 10, 24);

              ctx.fillStyle = '#ffffff';
              ctx.font = 'bold 12px Inter, sans-serif';
              ctx.fillText(point.label, x + 17, y + 4);
            }
          });
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, [detections, points, mode, videoRef]);

  if (mode === 'none') return null;

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
      style={{
        mixBlendMode: 'normal',
      }}
    />
  );
}
