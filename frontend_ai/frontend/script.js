const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const video = document.getElementById('webcam');
const emotionLabel = document.getElementById('emotion-label');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let model = null;  // You can remove blazeface stuff if not using it
let stream = null;
let animationId = null;
let detectionInterval = null;



async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

async function captureAndDetectEmotion() {
  if (!stream) return;

  // Create a temporary canvas for capturing video frame
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tempCtx = tempCanvas.getContext('2d');
  
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

  // Get base64 image data URL
  const base64Image = tempCanvas.toDataURL('image/png');

  try {
    const response = await fetch('http://127.0.0.1:5000/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image }),
    });

    if (!response.ok) {
      emotionLabel.textContent = 'Error detecting emotion';
      return;
    }

    const result = await response.json();
    emotionLabel.textContent = `Emotion: ${result.emotion}`;
    // Optional: show or open music link:
    console.log('Music link:', result.music);

  } catch (error) {
    emotionLabel.textContent = 'Backend connection error';
  }
}

function startDetectionLoop() {
  if (detectionInterval) return;  // prevent multiple intervals
  detectionInterval = setInterval(captureAndDetectEmotion, 2000); // every 2 seconds
}

function stopDetectionLoop() {
  if (detectionInterval) {
    clearInterval(detectionInterval);
    detectionInterval = null;
  }
}

async function start() {
  if (stream) return; // Already running
  await setupCamera();
  startDetectionLoop();
}

function stop() {
  if (!stream) return;
  stream.getTracks().forEach(track => track.stop());
  video.pause();
  video.srcObject = null;
  stream = null;
  emotionLabel.textContent = 'Detection stopped.';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  stopDetectionLoop();
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);


