// Handles webcam capture & upload detection

const webcamEl = document.getElementById("webcam");
const annotatedImg = document.getElementById("annotatedImg");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const intervalInput = document.getElementById("interval");
const uploadForm = document.getElementById("uploadForm");
const uploadResult = document.getElementById("uploadResult");

let stream = null;
let captureInterval = null;
const captureCanvas = document.createElement("canvas");
const ctx = captureCanvas.getContext("2d");

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamEl.srcObject = stream;
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    captureCanvas.width = settings.width || 640;
    captureCanvas.height = settings.height || 480;
    scheduleCapture();
  } catch (err) {
    alert("Error accessing webcam: " + err);
  }
}

function stopWebcam() {
  if (captureInterval) clearInterval(captureInterval);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

function scheduleCapture() {
  if (captureInterval) clearInterval(captureInterval);
  const ms = Math.max(100, parseInt(intervalInput.value) || 500);
  captureInterval = setInterval(captureAndSend, ms);
}

async function captureAndSend() {
  if (!stream) return;
  ctx.drawImage(webcamEl, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataURL = captureCanvas.toDataURL("image/jpeg", 0.7);
  try {
    const res = await fetch("/detect_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
    });
    if (!res.ok) return console.error("Server error:", res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    annotatedImg.src = url;
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  } catch (err) {
    console.error(err);
  }
}

startBtn.addEventListener("click", () => (!stream ? startWebcam() : scheduleCapture()));
stopBtn.addEventListener("click", stopWebcam);
intervalInput.addEventListener("change", scheduleCapture);

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("imageInput");
  if (!fileInput.files.length) {
    alert("Please choose an image.");
    return;
  }
  const formData = new FormData();
  formData.append("image", fileInput.files[0]); // key name MUST match Flask route
  const res = await fetch("/detect_upload", { method: "POST", body: formData });
  if (!res.ok) {
    const text = await res.text();
    alert("Upload failed!\n" + text);
    console.error("Server response:", text);
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  uploadResult.src = url;
  setTimeout(() => URL.revokeObjectURL(url), 3000);
});
// ================== HISTORY SECTION ==================
const viewHistoryBtn = document.getElementById("viewHistoryBtn");
const historySection = document.getElementById("historySection");
const historyGrid = document.getElementById("historyGrid");

viewHistoryBtn.addEventListener("click", async () => {
  const res = await fetch("/get_detected_images");
  if (!res.ok) {
    alert("Failed to fetch detected images!");
    return;
  }
  const data = await res.json();
  const images = data.images || [];
  historyGrid.innerHTML = "";
  if (images.length === 0) {
    historyGrid.innerHTML = "<p>No detected images yet.</p>";
  } else {
    images.forEach(url => {
      const img = document.createElement("img");
      img.src = url;
      img.alt = "Detected Image";
      img.className = "history-img";
      historyGrid.appendChild(img);
    });
  }
  historySection.classList.toggle("hidden");
});


window.addEventListener("beforeunload", stopWebcam);
