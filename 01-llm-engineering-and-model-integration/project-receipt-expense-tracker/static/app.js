let activeFile = null;
let currentCurrency = "$";

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
  const themeToggle = document.getElementById("theme-toggle");

  // Drag & drop
  dropZone.addEventListener("click", () => fileInput.click());
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
        showToast("Image pasted from clipboard!");
      }
    }
  });

  btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  btnExtract.addEventListener("click", performExtraction);

  // Filter & search
  document.getElementById("search-input").addEventListener("input", debounce(loadExpenses, 300));
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
    showToast(`Switched to ${nextTheme} mode`);
  });
}

function handleSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Please select a valid image file (JPG, PNG, WebP).");
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
  document.getElementById("scan-laser").classList.add("hidden");
}

async function performExtraction() {
  if (!activeFile) return;

  const btnExtract = document.getElementById("btn-extract");
  const laser = document.getElementById("scan-laser");
  const valBadge = document.getElementById("validation-badge");

  btnExtract.disabled = true;
  laser.classList.remove("hidden");
  valBadge.className = "badge badge-scanning";
  valBadge.textContent = "Scanning with Gemini Vision...";

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
      throw new Error(err.detail || "Failed to process receipt");
    }

    const data = await res.json();
    displayExtractionResult(data);
    loadStats();
    loadExpenses();
    showToast(`Successfully extracted ${data.merchant_name}!`);
  } catch (error) {
    console.error(error);
    showToast(`Extraction error: ${error.message}`);
    valBadge.className = "badge badge-idle";
    valBadge.textContent = "Extraction Failed";
  } finally {
    btnExtract.disabled = false;
    laser.classList.add("hidden");
  }
}

function displayExtractionResult(data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  const resultDisplay = document.getElementById("result-display");
  resultDisplay.classList.remove("hidden");

  document.getElementById("res-merchant").textContent = data.merchant_name || "Unknown Store";
  document.getElementById("res-date").textContent = `Purchase Date: ${data.purchase_date || "N/A"}`;

  // Category
  const catBadge = document.getElementById("res-category");
  catBadge.textContent = data.category || "other";
  catBadge.className = `category-badge cat-${(data.category || "other").toLowerCase()}`;

  // Currency
  currentCurrency = data.currency === "USD" ? "$" : `${data.currency} `;

  // Line items
  const tbody = document.getElementById("items-tbody");
  tbody.innerHTML = "";
  if (data.line_items && data.line_items.length > 0) {
    data.line_items.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(item.description)}</td>
        <td class="text-center font-mono">${item.quantity}</td>
        <td class="text-right font-mono">${currentCurrency}${item.unit_price.toFixed(2)}</td>
        <td class="text-right font-mono">${currentCurrency}${item.line_total.toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });
  } else {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted">No individual items extracted</td></tr>`;
  }

  // Totals
  document.getElementById("res-subtotal").textContent = `${currentCurrency}${data.subtotal.toFixed(2)}`;
  document.getElementById("res-tax").textContent = `${currentCurrency}${data.tax.toFixed(2)}`;
  document.getElementById("res-total").textContent = `${currentCurrency}${data.total.toFixed(2)}`;

  // Guardrail banner
  const banner = document.getElementById("guardrail-banner");
  const title = document.getElementById("guardrail-title");
  const desc = document.getElementById("guardrail-desc");
  const valBadge = document.getElementById("validation-badge");

  if (data.arithmetic_valid) {
    banner.className = "guardrail-banner banner-success";
    banner.querySelector(".banner-icon").textContent = "✓";
    title.textContent = "Deterministic Guardrail: Verified";
    desc.textContent = "Sum of extracted line items and tax matches printed grand total.";
    valBadge.className = "badge badge-success";
    valBadge.textContent = "Verified Schema";
  } else {
    banner.className = "guardrail-banner banner-warning";
    banner.querySelector(".banner-icon").textContent = "⚠️";
    title.textContent = `Arithmetic Discrepancy: ${currentCurrency}${data.discrepancy_amount.toFixed(2)}`;
    desc.textContent = "Mismatch between line items + tax and total. Flagged for manual review.";
    valBadge.className = "badge badge-idle";
    valBadge.textContent = "Review Needed";
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
    if (!res.ok) throw new Error("Failed to fetch expenses");
    const records = await res.json();

    tbody.innerHTML = "";
    if (records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="text-center empty-state">No receipts found. Upload your first receipt above!</td></tr>`;
      return;
    }

    records.forEach((r) => {
      const tr = document.createElement("tr");
      const thumb = r.image_url ? `<img src="${r.image_url}" alt="Receipt" class="receipt-thumb">` : "📄";
      const guardrailBadge = r.arithmetic_valid
        ? `<span class="guardrail-pill pill-pass">Passed</span>`
        : `<span class="guardrail-pill pill-mismatch">Mismatch ($${r.discrepancy_amount.toFixed(2)})</span>`;

      tr.innerHTML = `
        <td>${thumb}</td>
        <td class="font-mono">${r.purchase_date}</td>
        <td><strong>${escapeHtml(r.merchant_name)}</strong></td>
        <td><span class="category-badge cat-${r.category.toLowerCase()}">${r.category}</span></td>
        <td class="text-center font-mono">${r.line_items ? r.line_items.length : 0}</td>
        <td class="text-right font-mono">$${r.subtotal.toFixed(2)}</td>
        <td class="text-right font-mono">$${r.tax.toFixed(2)}</td>
        <td class="text-right font-mono"><strong>$${r.total.toFixed(2)}</strong></td>
        <td class="text-center">${guardrailBadge}</td>
        <td class="text-center">
          <button class="btn btn-sm btn-danger" onclick="deleteExpenseItem(${r.id})">Delete</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">Error loading ledger records.</td></tr>`;
  }
}

async function deleteExpenseItem(id) {
  if (!confirm("Are you sure you want to delete this expense record?")) return;
  try {
    const res = await fetch(`/api/expenses/${id}`, { method: "DELETE" });
    if (res.ok) {
      showToast("Expense record deleted.");
      loadStats();
      loadExpenses();
    }
  } catch (err) {
    showToast("Error deleting item.");
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
      pill.className = "status-pill status-live";
      text.textContent = `Gemini Live (${cfg.model})`;
    } else {
      pill.className = "status-pill status-demo";
      text.textContent = "Demo / Mock Mode (Click Settings)";
    }
  } catch (err) {
    console.warn("Could not check config", err);
  }
}

async function saveSettings() {
  const apiKey = document.getElementById("input-api-key").value;
  const model = document.getElementById("input-model-name").value;

  const formData = new FormData();
  formData.append("api_key", apiKey);
  formData.append("model", model);

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      body: formData
    });
    if (res.ok) {
      document.getElementById("modal-settings").classList.add("hidden");
      showToast("Configuration saved!");
      checkConfiguration();
    }
  } catch (err) {
    showToast("Failed to save settings.");
  }
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => {
    toast.classList.add("hidden");
  }, 3500);
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
