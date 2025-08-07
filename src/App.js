import React, { useState, useRef, useCallback, useEffect } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";
import Webcam from "react-webcam";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const [cameraMode, setCameraMode] = useState(false);
  const [facingMode, setFacingMode] = useState("environment");
  const [status, setStatus] = useState({ message: "Initializing camera...", type: "loading" });
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionCounts, setDetectionCounts] = useState({}); // Track detections by class
  
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const webcamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  // Configs
  const modelName = "oysters.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 300;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;
  
  function setupCanvasResize(canvas) {
    function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }

  // Modified detectImage wrapper to capture detection results
  const detectWithCounts = useCallback(async (imageSrc, canvas, session, topk, iouThreshold, scoreThreshold, modelInputShape) => {
    try {
      // Call the original detectImage function (now returns detections)
      var mode = 0;
      const detections = await detectImage(
        imageSrc,
        canvas,
        session,
        topk,
        iouThreshold,
        scoreThreshold,
        modelInputShape, mode
      );
      
      // Process detection data for counting
      if (detections && Array.isArray(detections)) {
        const counts = {};
        detections.forEach(detection => {
          const className = detection.class || 'Unknown';
          counts[className] = (counts[className] || 0) + 1;
        });
        setDetectionCounts(counts);
      } else {
        setDetectionCounts({});
      }
    } catch (error) {
      console.error('Detection error:', error);
      setDetectionCounts({});
    }
  }, []);

  // Process frame for detection
  const processFrame = useCallback(() => {
    if (!webcamRef.current || !session || !canvasRef.current) return;
    
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    const img = new Image();
    img.onload = () => {
      detectWithCounts(
        img,
        canvasRef.current,
        session,
        topk,
        iouThreshold,
        scoreThreshold,
        modelInputShape
      );
    };
    img.src = imageSrc;
  }, [session, topk, iouThreshold, scoreThreshold, modelInputShape, detectWithCounts]);

  // Start continuous detection
  const startDetection = useCallback(() => {
    if (isDetecting) return;
    
    setIsDetecting(true);
    console.log('Starting detection loop...');
    
    const detectLoop = () => {
      processFrame();
      detectionIntervalRef.current = requestAnimationFrame(detectLoop);
    };
    
    detectLoop();
  }, [isDetecting, processFrame]);

  // Stop continuous detection
  const stopDetection = useCallback(() => {
    if (detectionIntervalRef.current) {
      cancelAnimationFrame(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    setIsDetecting(false);
    setDetectionCounts({});
    console.log('Stopped detection loop');
  }, []);

  // Enable camera mode
  const enableCameraMode = useCallback(() => {
    setCameraMode(true);
    setStatus({ message: "Initializing camera...", type: "loading" });
  }, []);

  // Exit camera mode
  const exitCameraMode = useCallback(() => {
    stopDetection();
    setCameraMode(false);
    setImage(null);
    setDetectionCounts({});
  }, [stopDetection]);

  // Flip camera
  const flipCamera = useCallback(() => {
    stopDetection();
    setFacingMode(prev => prev === "user" ? "environment" : "user");
    setStatus({ message: "Switching camera...", type: "loading" });
  }, [stopDetection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, [stopDetection]);

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`,
      ["Loading YOLOv8 Segmentation model", setLoading]
    );
    const yolov8 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov8.onnx`,
      ["Loading NMS model", setLoading]
    );
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms });
    setLoading(null);
    const myCanvas = document.getElementById('canvas');
    setupCanvasResize(myCanvas);
  };

  // Calculate total detections
  const getTotalDetections = () => {
    return Object.values(detectionCounts).reduce((sum, count) => sum + count, 0);
  };

  // Camera mode UI
  if (cameraMode) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: '#000',
        overflow: 'hidden'
      }}>
        {/* Detection counts display */}
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.8)',
          padding: '15px 20px',
          borderRadius: '10px',
          color: 'white',
          zIndex: 11,
          minWidth: '200px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>
            Detections: {getTotalDetections()}
          </div>
          {Object.keys(detectionCounts).length > 0 && (
            <div style={{ fontSize: '14px', lineHeight: '1.5' }}>
              {Object.entries(detectionCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([className, count]) => (
                  <div key={className} style={{ 
                    padding: '2px 0',
                    borderTop: '1px solid rgba(255,255,255,0.2)',
                    marginTop: '4px',
                    paddingTop: '6px'
                  }}>
                    {className}: {count}
                  </div>
                ))}
            </div>
          )}
          {Object.keys(detectionCounts).length === 0 && (
            <div style={{ fontSize: '12px', opacity: 0.7 }}>
              No objects detected
            </div>
          )}
        </div>

        {/* Camera View */}
        <div style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.92}
            videoConstraints={{
              facingMode: facingMode,
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            }}
            onUserMedia={() => {
              console.log('Webcam ready');
              setStatus({ message: "Camera ready", type: "success" });
              if (session && !isDetecting) {
                console.log('Starting detection after webcam ready');
                setTimeout(() => startDetection(), 500);
              }
            }}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
          
          {/* Detection overlay canvas */}
          <canvas
            id="canvas"
            ref={canvasRef}
            width={window.innerWidth}
            height={window.innerHeight}
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 'auto',
              height: 'auto',
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              pointerEvents: 'none'
            }}
          />
        </div>

        {/* Status bar */}
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          right: '20px',
          textAlign: 'center',
          background: 'rgba(0,0,0,0.7)',
          padding: '10px',
          borderRadius: '10px',
          fontSize: '14px',
          color: status.type === 'error' ? '#ff4444' : 
                 status.type === 'success' ? '#44ff44' : 
                 status.type === 'loading' ? '#ffaa44' : 'white',
          zIndex: 10
        }}>
          {status.message}
        </div>

        {/* Exit button */}
        <button
          onClick={exitCameraMode}
          style={{
            position: 'absolute',
            top: '20px',
            left: '20px',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            border: '2px solid white',
            background: 'rgba(0,0,0,0.5)',
            color: 'white',
            cursor: 'pointer',
            fontSize: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 11
          }}
        >
          ✕
        </button>

        {/* Flip camera button */}
        <button
          onClick={flipCamera}
          title="Switch Camera"
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            width: '50px',
            height: '50px',
            borderRadius: '50%',
            border: '2px solid white',
            background: 'rgba(0,0,0,0.5)',
            color: 'white',
            cursor: 'pointer',
            fontSize: '24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 11
          }}
        >
          🔄
        </button>
      </div>
    );
  }

  // Original app UI
  return (
    <div className="App">
      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          pointerEvents: loading ? 'auto' : 'none'
        }}>
          <Loader>
            {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
          </Loader>
        </div>
      )}
      <div className="header">
      </div>

      <div className="content">
        <button 
          onClick={enableCameraMode}
          disabled={!session}
          style={{
            padding: '15px 30px',
            fontSize: '18px',
            background: session ? '#007bff' : '#cccccc',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: session ? 'pointer' : 'not-allowed',
            marginBottom: '20px',
            position: 'relative',
            zIndex: 1
          }}
        >
          {session ? 'Start YOLO Application' : 'Loading Models...'}
        </button>
        
        {/* Show detection counts for uploaded images */}
        {image && !cameraMode && Object.keys(detectionCounts).length > 0 && (
          <div style={{
            background: 'rgba(0,0,0,0.8)',
            padding: '15px',
            borderRadius: '10px',
            color: 'white',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>
              Detections: {getTotalDetections()}
            </div>
            <div style={{ fontSize: '14px' }}>
              {Object.entries(detectionCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([className, count]) => (
                  <div key={className} style={{ padding: '2px 0' }}>
                    {className}: {count}
                  </div>
                ))}
            </div>
          </div>
        )}
        </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image && !cameraMode ? "block" : "none" }}
          onLoad={async () => {
            if (!cameraMode && session && canvasRef.current) {
              await detectWithCounts(
                imageRef.current,
                canvasRef.current,
                session,
                topk,
                iouThreshold,
                scoreThreshold,
                modelInputShape
              );
            }
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
          style={{ display: !cameraMode ? 'block' : 'none' }}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]);
          imageRef.current.src = url;
          setImage(url);
        }}
      />
      
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
          disabled={!session}
          style={{
            opacity: session ? 1 : 0.5,
            cursor: session ? 'pointer' : 'not-allowed'
          }}
        >
          {session ? 'Open local image' : 'Loading...'}
        </button>
        {image && !cameraMode && (
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
              setDetectionCounts({});
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;