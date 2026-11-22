const API_URL = window.location.origin;

const videoInput = document.querySelector("#videoInput");
const dropzone = document.querySelector("#dropzone");
const fileName = document.querySelector("#fileName");
const fileSize = document.querySelector("#fileSize");
const processButton = document.querySelector("#processButton");
const error = document.querySelector("#error");
const results = document.querySelector("#results");
const emptyResult = document.querySelector("#emptyResult");
const beforeVideo = document.querySelector("#beforeVideo");
const afterVideo = document.querySelector("#afterVideo");
const downloadLink = document.querySelector("#downloadLink");
const metrics = document.querySelector("#metrics");
const pipelineNote = document.querySelector("#pipelineNote");

let selectedFile = null;

function stagesProcessing() {
  const stages = [...document.querySelectorAll(".stage")];
  stages.forEach((stage, index) => {
    stage.classList.toggle("active", index === 0);
  });
  let index = 0;
  const timer = window.setInterval(() => {
    stages[index]?.classList.remove("active");
    stages[index]?.classList.add("complete");
    index += 1;
    if (index < stages.length) {
      stages[index]?.classList.add("active");
    } else {
      window.clearInterval(timer);
    }
  }, 500);
}

function selectFile(file) {
  if (!file) return;

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  processButton.disabled = false;
  processButton.textContent = "Blur & protect video";
  error.hidden = true;
  results.hidden = true;
  emptyResult.hidden = false;
  downloadLink.hidden = true;
  document.querySelectorAll(".stage").forEach((stage) => stage.classList.remove("complete"));
  beforeVideo.src = URL.createObjectURL(file);
}

videoInput.addEventListener("change", (event) => {
  selectFile(event.target.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  selectFile(event.dataTransfer.files[0]);
});

processButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  processButton.disabled = true;
  processButton.innerHTML = '<span class="spinner"></span> Processing video';
  error.hidden = true;
  results.hidden = true;
  emptyResult.hidden = false;
  stagesProcessing();

  const form = new FormData();
  form.append("video", selectedFile);

  try {
    let response;
    try {
      response = await fetch(`${API_URL}/api/process`, {
        method: "POST",
        body: form,
      });
    } catch (networkError) {
      throw new Error(
        `Cannot reach the backend at ${window.location.origin}. Restart Pixora Shield on port 3000.`
      );
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Backend returned HTTP ${response.status} without JSON.`);
    }

    if (!response.ok) {
      throw new Error(data.error || "Processing failed");
    }

    const processedUrl = `${API_URL}${data.processed_url}`;
    afterVideo.src = processedUrl;
    downloadLink.href = `${API_URL}/api/download/${data.processed_url.split("/").pop()}`;

    metrics.innerHTML = [
      [data.metrics.frames, "Frames processed"],
      [data.metrics.faces_and_plates_detected, "Privacy regions found"],
      [`${data.metrics.average_fps} fps`, "Average processing"],
      [data.metrics.resolution, "Resolution"],
    ].map(([value, label]) =>
      `<div><strong>${value}</strong><span>${label}</span></div>`
    ).join("");

    pipelineNote.textContent = data.pipeline_summary || "Privacy regions have been blurred in the processed video.";
    emptyResult.hidden = true;
    downloadLink.hidden = false;
    results.hidden = false;
    document.querySelectorAll(".stage").forEach((stage) => stage.classList.add("complete"));
  } catch (caught) {
    error.textContent = caught instanceof Error ? caught.message : "Processing failed";
    error.hidden = false;
  } finally {
    processButton.disabled = false;
    processButton.textContent = "Blur & protect video";
  }
});
