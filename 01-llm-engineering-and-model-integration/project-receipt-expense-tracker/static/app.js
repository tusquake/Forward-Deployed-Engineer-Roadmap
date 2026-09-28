let activeFile = null;
let currentCurrency = "$";

const SVG_CHECK = `
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
    <polyline points="22 4 12 14.01 9 11.01"></polyline>
  </svg>
`;

const SVG_ALERT = `
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
    <line x1="12" y1="9" x2="12" y2="13"></line>
    <line x1="12" y1="17" x2="12.01" y2="17"></line>
  </svg>
`;

const SVG_DOC_PLACEHOLDER = `
  <div class="thumb-placeholder" title="No image uploaded">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
      <polyline points="14 2 14 8 20 8"></polyline>
    </svg>
  </div>
`;

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  checkConfiguration();
  loadStats();
  loadExpenses();
});

function initEventListeners() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnRemove = document.getElementById("btn-remove-image");
  const btnExtract = document.getElementById("btn-extract");
  const btnSample = document.getElementById("btn-sample-receipt");
  const themeToggle = document.getElementById("theme-toggle");

  // Drag & drop
  dropZone.addEventListener("click", (e) => {
    if (e.target.closest("#btn-remove-image")) return;
    fileInput.click();
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  // Paste from clipboard
  window.addEventListener("paste", (e) => {
    if (e.clipboardData && e.clipboardData.files.length > 0) {
      const file = e.clipboardData.files[0];
      if (file.type.startsWith("image/")) {
        handleSelectedFile(file);
        showToast("Image pasted from clipboard");
      }
    }
  });

  // Remove preview
  btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  // Load sample receipt button
  if (btnSample) {
    btnSample.addEventListener("click", loadSampleReceipt);
  }

  // Extract action
  btnExtract.addEventListener("click", performExtraction);

  // Filter & search
  document.getElementById("search-input").addEventListener("input", debounce(loadExpenses, 250));
  document.getElementById("filter-category").addEventListener("change", loadExpenses);

  // Settings modal
  const modal = document.getElementById("modal-settings");
  document.getElementById("btn-settings").addEventListener("click", () => {
    modal.classList.remove("hidden");
  });
  document.getElementById("btn-close-modal").addEventListener("click", () => {
    modal.classList.add("hidden");
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  document.getElementById("btn-save-settings").addEventListener("click", saveSettings);

  // Theme toggle
  themeToggle.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme") || "dark";
    const nextTheme = current === "dark" ? "light" : "dark";
    document.body.setAttribute("data-theme", nextTheme);
    showToast(`Switched to ${nextTheme} theme`);
  });
}

async function loadSampleReceipt() {
  try {
    const res = await fetch("/sample_whole_foods.png");
    if (!res.ok) throw new Error("Sample receipt file not accessible");
    const blob = await res.blob();
    const file = new File([blob], "sample_whole_foods.png", { type: "image/png" });
    handleSelectedFile(file);
    showToast("Loaded sample receipt: Whole Foods Market");
  } catch (err) {
    console.error(err);
    showToast("Could not load sample receipt");
  }
}

function handleSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Please provide a valid image (PNG, JPG, or WebP)");
    return;
  }

  activeFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("image-preview").src = e.target.result;
    document.getElementById("drop-prompt").classList.add("hidden");
    document.getElementById("preview-container").classList.remove("hidden");
    document.getElementById("btn-extract").disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearSelectedFile() {
  activeFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("image-preview").src = "";
  document.getElementById("preview-container").classList.add("hidden");
  document.getElementById("drop-prompt").classList.remove("hidden");
  document.getElementById("btn-extract").disabled = true;
  document.getElementById("processing-bar").classList.add("hidden");
}

async function performExtraction() {
  if (!activeFile) return;

  const btnExtract = document.getElementById("btn-extract");
  const btnLabel = document.getElementById("btn-extract-label");
  const progressBar = document.getElementById("processing-bar");
  const valBadge = document.getElementById("validation-badge");

  btnExtract.disabled = true;
  btnLabel.textContent = "Extracting...";
  progressBar.classList.remove("hidden");

  valBadge.className = "status-tag tag-active";
  valBadge.textContent = "Processing";

  const formData = new FormData();
  formData.append("file", activeFile);
  const selectedModel = document.getElementById("select-model").value;
  formData.append("custom_model", selectedModel);

  try {
    const res = await fetch("/api/extract", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Extraction failed");
    }

    const data = await res.json();
    displayExtractionResult(data);
    loadStats();
    loadExpenses();
    showToast(`Extracted: ${data.merchant_name}`);
  } catch (error) {
    console.error(error);
    showToast(`Extraction error: ${error.message}`);
    valBadge.className = "status-tag tag-warning";
    valBadge.textContent = "Failed";
  } finally {
    btnExtract.disabled = false;
    btnLabel.textContent = "Process Receipt";
    progressBar.classList.add("hidden");
  }
}

function displayExtractionResult(data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const resultDisplay = document.getElementById("result-display");
  resultDisplay.classList.remove("hidden");

  document.getElementById("res-merchant").textContent = data.merchant_name || "Unidentified Merchant";
  document.getElementById("res-date").textContent = `Purchase Date: ${data.purchase_date || "N/A"}`;

  // Category
  const catBadge = document.getElementById("res-category");
  catBadge.textContent = data.category || "other";
  catBadge.className = `category-pill cat-${(data.category || "other").toLowerCase()}`;

  // Currency
  currentCurrency = data.currency === "USD" ? "$" : `${data.currency} `;

  // Line items table
  const tbody = document.getElementById("items-tbody");
  tbody.innerHTML = "";
  if (data.line_items && data.line_items.length > 0) {
    data.line_items.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(item.description)}</td>
        <td class="cell-center num-tabular">${item.quantity}</td>
        <td class="cell-right num-tabular">${currentCurrency}${item.unit_price.toFixed(2)}</td>
        <td class="cell-right num-tabular">${currentCurrency}${item.line_total.toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });
  } else {
    tbody.innerHTML = `<tr><td colspan="4" class="cell-center cell-muted">No individual line items parsed</td></tr>`;
  }

  // Totals
  document.getElementById("res-subtotal").textContent = `${currentCurrency}${data.subtotal.toFixed(2)}`;
  document.getElementById("res-tax").textContent = `${currentCurrency}${data.tax.toFixed(2)}`;
  document.getElementById("res-total").textContent = `${currentCurrency}${data.total.toFixed(2)}`;

  // Guardrail banner
  const banner = document.getElementById("guardrail-banner");
  const iconWrap = document.getElementById("guardrail-icon-wrap");
  const title = document.getElementById("guardrail-title");
  const desc = document.getElementById("guardrail-desc");
  const valBadge = document.getElementById("validation-badge");

  if (data.arithmetic_valid) {
    banner.className = "audit-card audit-pass";
    iconWrap.innerHTML = SVG_CHECK;
    title.textContent = "Arithmetic Reconciled";
    desc.textContent = "Sum of extracted line items and recorded tax matches the grand total.";
    valBadge.className = "status-tag tag-success";
    valBadge.textContent = "Reconciled";
  } else {
    banner.className = "audit-card audit-mismatch";
    iconWrap.innerHTML = SVG_ALERT;
    title.textContent = `Discrepancy: ${currentCurrency}${data.discrepancy_amount.toFixed(2)}`;
    desc.textContent = "Computed sum differs from printed grand total. Flagged for review.";
    valBadge.className = "status-tag tag-warning";
    valBadge.textContent = "Variance Flagged";
  }

  // Confidence Notes
  const confBox = document.getElementById("confidence-box");
  const confText = document.getElementById("res-notes");
  if (data.confidence_notes && data.confidence_notes.trim()) {
    confText.textContent = data.confidence_notes;
    confBox.classList.remove("hidden");
  } else {
    confBox.classList.add("hidden");
  }
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const stats = await res.json();
    document.getElementById("stat-total-spend").textContent = `$${stats.total_spend.toFixed(2)}`;
    document.getElementById("stat-total-count").textContent = stats.total_receipts;
    document.getElementById("stat-top-category").textContent = stats.top_category;
    document.getElementById("stat-flagged").textContent = stats.flagged_discrepancies;
  } catch (err) {
    console.warn("Could not load stats", err);
  }
}

async function loadExpenses() {
  const category = document.getElementById("filter-category").value;
  const search = document.getElementById("search-input").value;
  const tbody = document.getElementById("ledger-tbody");

  let url = "/api/expenses?";
  if (category && category !== "all") url += `category=${encodeURIComponent(category)}&`;
  if (search && search.trim()) url += `search=${encodeURIComponent(search.trim())}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load records");
    const records = await res.json();

    tbody.innerHTML = "";
    if (records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="cell-center cell-muted" style="padding: 24px;">No expense records found.</td></tr>`;
      return;
    }

    records.forEach((r) => {
      const tr = document.createElement("tr");
      const thumb = r.image_url
        ? `<img src="${r.image_url}" alt="Receipt" class="receipt-thumb">`
        : SVG_DOC_PLACEHOLDER;

      const guardrailBadge = r.arithmetic_valid
        ? `<span class="audit-badge-pill badge-reconciled">Reconciled</span>`
        : `<span class="audit-badge-pill badge-discrepancy">Variance ($${r.discrepancy_amount.toFixed(2)})</span>`;

      tr.innerHTML = `
        <td>${thumb}</td>
        <td class="num-tabular">${r.purchase_date}</td>
        <td><strong>${escapeHtml(r.merchant_name)}</strong></td>
        <td><span class="category-pill cat-${r.category.toLowerCase()}">${r.category}</span></td>
        <td class="cell-center num-tabular">${r.line_items ? r.line_items.length : 0}</td>
        <td class="cell-right num-tabular">$${r.subtotal.toFixed(2)}</td>
        <td class="cell-right num-tabular">$${r.tax.toFixed(2)}</td>
        <td class="cell-right num-tabular"><strong>$${r.total.toFixed(2)}</strong></td>
        <td class="cell-center">${guardrailBadge}</td>
        <td class="cell-center">
          <button class="btn btn-xs btn-danger" onclick="deleteExpenseItem(${r.id})" title="Delete record" type="button">
            Delete
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="10" class="cell-center cell-muted">Failed to load ledger records.</td></tr>`;
  }
}

async function deleteExpenseItem(id) {
  if (!confirm("Are you sure you want to remove this expense record?")) return;
  try {
    const res = await fetch(`/api/expenses/${id}`, { method: "DELETE" });
    if (res.ok) {
      showToast("Expense record removed");
      loadStats();
      loadExpenses();
    }
  } catch (err) {
    showToast("Error deleting record");
  }
}

async function checkConfiguration() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    const cfg = await res.json();
    const pill = document.getElementById("status-pill");
    const text = document.getElementById("status-text");

    if (cfg.is_configured) {
      pill.className = "status-indicator status-ready";
      if (cfg.has_gemini && cfg.has_groq_fallback) {
        text.textContent = `Online: Gemini + Groq Fallback`;
      } else if (cfg.has_gemini) {
        text.textContent = `Online: Gemini (${cfg.model})`;
      } else {
        text.textContent = `Online: Groq Fallback`;
      }
    } else {
      pill.className = "status-indicator status-warning";
      text.textContent = "Offline (Configure in Settings)";
    }

    // Set model in input if present
    const modelInput = document.getElementById("input-model-name");
    if (modelInput && cfg.model) {
      modelInput.value = cfg.model;
    }
  } catch (err) {
    console.warn("Could not check config", err);
  }
}

async function saveSettings() {
  const apiKey = document.getElementById("input-api-key").value;
  const groqKey = document.getElementById("input-groq-key").value;
  const model = document.getElementById("input-model-name").value;

  const formData = new FormData();
  if (apiKey.trim()) formData.append("api_key", apiKey.trim());
  if (groqKey.trim()) formData.append("groq_key", groqKey.trim());
  if (model.trim()) formData.append("model", model.trim());

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      body: formData
    });
    if (res.ok) {
      document.getElementById("modal-settings").classList.add("hidden");
      showToast("Settings updated successfully");
      checkConfiguration();
    }
  } catch (err) {
    showToast("Failed to save settings");
  }
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => {
    toast.classList.add("hidden");
  }, 3200);
}

function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
