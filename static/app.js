const form = document.querySelector("#query-form");
const uploadForm = document.querySelector("#upload-form");
const question = document.querySelector("#question");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const evidence = document.querySelector("#evidence");
const statusText = document.querySelector("#status");
const reindexButton = document.querySelector("#reindex-button");
const paperList = document.querySelector("#paper-list");
const paperCount = document.querySelector("#paper-count");

function setLoading(isLoading, label = "Ready") {
  statusText.textContent = label;
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isLoading;
  });
}

function renderSources(items) {
  sources.innerHTML = "";

  if (!items || items.length === 0) {
    return;
  }

  items.forEach((item) => {
    const pill = document.createElement("span");
    pill.className = "source-pill";
    pill.textContent = item;
    sources.appendChild(pill);
  });
}

function renderEvidence(items) {
  evidence.innerHTML = "";

  if (!items || items.length === 0) {
    return;
  }

  const title = document.createElement("h3");
  title.textContent = "Retrieved evidence";
  evidence.appendChild(title);

  items.forEach((item) => {
    const block = document.createElement("article");
    block.className = "evidence-item";

    const source = document.createElement("strong");
    source.textContent = `${item.title}${Number.isInteger(item.page) ? `, page ${item.page + 1}` : ""}`;

    const preview = document.createElement("p");
    preview.textContent = `Section: ${item.section}`;

    block.append(source, preview);
    evidence.appendChild(block);
  });
}

function renderPapers(items) {
  paperCount.textContent = items.length;
  paperList.innerHTML = "";
  paperList.classList.toggle("empty", items.length === 0);

  if (items.length === 0) {
    paperList.textContent = "No papers loaded yet.";
    return;
  }

  items.forEach((paper) => {
    const item = document.createElement("div");
    item.className = "paper-item";
    const title = document.createElement("strong");
    title.textContent = paper.title || paper.name;
    const filename = document.createElement("span");
    filename.textContent = paper.name;
    item.append(title, filename);
    paperList.appendChild(item);
  });
}

async function loadPapers() {
  const response = await fetch("/api/papers");
  const data = await response.json();
  renderPapers(data.papers || []);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const value = question.value.trim();
  if (!value) {
    answer.textContent = "Please enter a research question.";
    answer.classList.remove("empty");
    return;
  }

  setLoading(true, "Thinking");
  answer.textContent = "Searching papers and asking Granite...";
  answer.classList.remove("empty");
  renderSources([]);
  renderEvidence([]);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Research query failed.");
    }

    answer.textContent = data.answer;
    renderSources(data.sources);
    renderEvidence(data.evidence);
    setLoading(false, "Complete");
  } catch (error) {
    answer.textContent = error.message;
    setLoading(false, "Error");
  }
});

reindexButton.addEventListener("click", async () => {
  setLoading(true, "Indexing");
  answer.textContent = "Rebuilding the FAISS vectorstore from PDFs...";
  answer.classList.remove("empty");
  renderSources([]);
  renderEvidence([]);

  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Reindexing failed.");
    }

    answer.textContent = data.message;
    renderPapers(data.papers || []);
    setLoading(false, "Indexed");
  } catch (error) {
    answer.textContent = error.message;
    setLoading(false, "Error");
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(uploadForm);
  if (!formData.getAll("papers").some((file) => file && file.name)) {
    answer.textContent = "Choose at least one PDF file to upload.";
    answer.classList.remove("empty");
    return;
  }

  setLoading(true, "Uploading");
  answer.textContent = "Uploading PDFs and rebuilding the FAISS index...";
  answer.classList.remove("empty");
  renderSources([]);
  renderEvidence([]);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }

    answer.textContent = data.message;
    renderPapers(data.papers || []);
    uploadForm.reset();
    setLoading(false, "Indexed");
  } catch (error) {
    answer.textContent = error.message;
    setLoading(false, "Error");
  }
});

loadPapers().catch(() => {
  paperList.textContent = "Could not load paper list.";
});
