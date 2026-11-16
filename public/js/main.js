/**
 * Traffic Sign Recognition — Frontend Application Controller
 * Handles drag-and-drop, preview rendering, sample selection, and real-time inference.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const previewBox = document.getElementById("preview-box");
  const previewImg = document.getElementById("preview-img");
  const previewName = document.getElementById("preview-name");
  const previewDims = document.getElementById("preview-dims");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-btn");
  const samplesContainer = document.getElementById("samples-container");

  // Result Elements
  const resultsPanel = document.getElementById("results-panel");
  const resultImg = document.getElementById("result-img");
  const predictedName = document.getElementById("predicted-name");
  const predictedClassId = document.getElementById("predicted-class-id");
  const confidencePct = document.getElementById("confidence-pct");
  const confidenceBar = document.getElementById("confidence-bar");
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const rankingList = document.getElementById("ranking-list");
  const latencyText = document.getElementById("latency-text");
  const resetBtn = document.getElementById("reset-btn");
  const modelParamCount = document.getElementById("model-param-count");

  // State
  let currentFile = null;
  let currentSampleId = null;
  let currentPreviewUrl = null;

  // 1. Fetch Sample Signs
  fetch("/api/samples")
    .then((res) => res.json())
    .then((samples) => {
      if (!samplesContainer) return;
      samplesContainer.innerHTML = "";

      if (samples.length === 0) {
        samplesContainer.innerHTML = `<span style="font-size:12px;color:var(--text-muted);">No pre-loaded samples found. Upload any sign image above.</span>`;
        return;
      }

      samples.forEach((sample) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sample-chip";
        chip.innerHTML = `
          <img src="${sample.url}" alt="${sample.name}">
          <span>${sample.name}</span>
        `;
        chip.addEventListener("click", () => selectSample(sample));
        samplesContainer.appendChild(chip);
      });
    })
    .catch((err) => console.warn("Samples fetch failed:", err));

  // 3. File Selection Handlers
  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  // Drag and Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-active");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      handleSelectedFile(dt.files[0]);
    }
  });

  function handleSelectedFile(file) {
    currentFile = file;
    currentSampleId = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      currentPreviewUrl = e.target.result;
      showPreview(file.name, currentPreviewUrl, `${Math.round(file.size / 1024)} KB`);
    };
    reader.readAsDataURL(file);
  }

  function selectSample(sample) {
    currentFile = null;
    currentSampleId = sample.id;
    currentPreviewUrl = sample.url;
    showPreview(sample.name, sample.url, "Standard GTSRB Test Sample");
  }

  function showPreview(name, url, metaText) {
    previewName.textContent = name;
    previewDims.textContent = metaText;
    previewImg.src = url;
    previewBox.style.display = "block";
    resultsPanel.style.display = "none";
  }

  // Clear / Change Selection
  clearBtn.addEventListener("click", resetAll);
  resetBtn.addEventListener("click", resetAll);

  function resetAll() {
    currentFile = null;
    currentSampleId = null;
    currentPreviewUrl = null;
    fileInput.value = "";
    previewBox.style.display = "none";
    resultsPanel.style.display = "none";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // 4. Run Analysis via FastAPI Backend
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile && !currentSampleId) return;

    // Loading UI state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon">
        <line x1="12" y1="2" x2="12" y2="6"></line>
        <line x1="12" y1="18" x2="12" y2="22"></line>
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
        <line x1="2" y1="12" x2="6" y2="12"></line>
        <line x1="18" y1="12" x2="22" y2="12"></line>
      </svg>
      <span>Processing...</span>
    `;

    try {
      let response;
      if (currentSampleId) {
        response = await fetch(`/api/predict?sample_id=${encodeURIComponent(currentSampleId)}`, {
          method: "POST",
        });
      } else {
        const formData = new FormData();
        formData.append("file", currentFile);
        response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Prediction request failed.");
      }

      const result = await response.json();
      displayResults(result);
    } catch (err) {
      alert(`Inference Error: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
        <span>Analyze Sign</span>
      `;
    }
  });

  // 5. Render Results
  function displayResults(data) {
    resultImg.src = currentPreviewUrl;
    predictedName.textContent = data.sign_name;
    predictedClassId.textContent = `GTSRB Class #${data.predicted_class}`;
    confidencePct.textContent = `${data.confidence.toFixed(2)}%`;
    confidenceBar.style.width = `${Math.min(100, data.confidence)}%`;

    // Dynamic color coding based on autonomous driving safety criteria
    if (data.confidence >= 90.0) {
      confidenceBar.style.backgroundColor = "var(--accent-green)";
      statusPill.className = "status-badge success";
      statusText.textContent = "High Certainty (Meets ADAS Safety Threshold)";
    } else if (data.confidence >= 70.0) {
      confidenceBar.style.backgroundColor = "var(--accent-blue)";
      statusPill.className = "status-badge";
      statusPill.style.backgroundColor = "var(--accent-blue-subtle)";
      statusPill.style.color = "var(--accent-blue)";
      statusPill.style.border = "1px solid var(--accent-blue-border)";
      statusText.textContent = "Moderate Confidence";
    } else {
      confidenceBar.style.backgroundColor = "var(--accent-red)";
      statusPill.className = "status-badge warning";
      statusText.textContent = "Low Confidence (Sensor Retest Required)";
    }

    if (latencyText) {
      latencyText.textContent = `Inference latency: ${data.latency_ms} ms (Real-time CPU)`;
    }

    // Top Predictions Distribution
    rankingList.innerHTML = "";
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "ranking-row";
        row.innerHTML = `
          <span class="ranking-name">${index + 1}. ${item.name}</span>
          <span class="ranking-pct">${item.confidence.toFixed(2)}%</span>
        `;
        rankingList.appendChild(row);
      });
    }

    resultsPanel.style.display = "block";
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
