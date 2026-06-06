import os
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from werkzeug.utils import secure_filename


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIRECTORY = BASE_DIR / os.getenv("PDF_DIRECTORY", "data/papers")
VECTORSTORE_DIRECTORY = BASE_DIR / os.getenv("VECTORSTORE_DIRECTORY", "vectorstore")

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

IMPORTANT_SECTIONS = {
    "title",
    "abstract",
    "introduction",
    "methodology",
    "methods",
    "approach",
    "model",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "future work",
}

LOW_VALUE_SECTIONS = {"references", "appendix", "acknowledgments"}

SECTION_PATTERNS = [
    ("abstract", r"\babstract\b"),
    ("introduction", r"\b(1\.?\s*)?introduction\b"),
    ("methodology", r"\b(methodology|methods?|approach|proposed method)\b"),
    ("model", r"\b(model|architecture)\b"),
    ("experiments", r"\b(experiments?|experimental setup|evaluation)\b"),
    ("results", r"\b(results?|analysis)\b"),
    ("discussion", r"\bdiscussion\b"),
    ("limitations", r"\b(limitations?|threats to validity)\b"),
    ("future work", r"\bfuture work\b"),
    ("conclusion", r"\b(conclusions?|concluding remarks)\b"),
    ("appendix", r"\b(appendix|supplementary)\b"),
    ("references", r"\b(references|bibliography)\b"),
]


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def clean_text(text):
    text = re.sub(r"-\s*\n\s*", "", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_title(pdf_path, first_page_text):
    lines = [
        clean_text(line)
        for line in (first_page_text or "").splitlines()
        if clean_text(line)
    ]
    blocked = {
        "abstract",
        "introduction",
        "references",
        "proceedings",
        "preprint",
        "permission",
        "copyright",
        "reproduce",
        "journalistic",
        "scholarly",
        "google hereby",
        "microsoft research",
        "openai",
    }
    candidates = []

    for line in lines[:18]:
        lower = line.lower()
        if any(word in lower for word in blocked):
            continue
        if len(line) < 8 or len(line) > 160:
            continue
        if re.search(r"@|www\.|http|arxiv|copyright|\{|\}", lower):
            continue
        if len(re.findall(r"\b[A-Z][a-z]+", line)) > 5 and not re.search(r"\b(is|are|via|with|for|all|you|need)\b", lower):
            continue
        candidates.append(line)

    if candidates:
        return candidates[0]

    try:
        reader = PdfReader(str(pdf_path))
        metadata_title = (reader.metadata or {}).get("/Title")
        if metadata_title and len(metadata_title.strip()) > 5:
            return clean_text(metadata_title)[:140]
    except Exception:
        pass

    return pdf_path.stem


def detect_section(text, page_number):
    lowered = text.lower()

    if page_number == 0:
        if "abstract" in lowered:
            return "abstract"
        return "title"

    for section, pattern in SECTION_PATTERNS:
        if re.search(pattern, lowered[:1800], flags=re.IGNORECASE):
            return section

    return "body"


def priority_for_section(section, page_number, text):
    priority = 1.0

    if page_number <= 2:
        priority += 1.0
    if section in IMPORTANT_SECTIONS:
        priority += 1.3
    if section in {"abstract", "introduction", "methodology", "methods", "approach", "conclusion", "future work"}:
        priority += 0.8
    if section in LOW_VALUE_SECTIONS:
        priority -= 1.5
    if text.count("|") > 8 or len(re.findall(r"\b\d+(\.\d+)?\b", text)) > 80:
        priority -= 0.6

    return max(priority, 0.2)


def extract_pdf_documents():
    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    documents = []
    page_count = 0

    for pdf_path in sorted(PDF_DIRECTORY.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        first_page_text = reader.pages[0].extract_text() if reader.pages else ""
        title = extract_title(pdf_path, first_page_text)

        overview_parts = []
        for page_index, page in enumerate(reader.pages):
            text = clean_text(page.extract_text())
            if not text:
                continue

            section = detect_section(text, page_index)
            priority = priority_for_section(section, page_index, text)
            page_count += 1

            if page_index <= 2 or section in {"abstract", "introduction", "methodology", "conclusion", "future work"}:
                overview_parts.append(f"[{section}, page {page_index + 1}] {text[:1200]}")

            documents.append(
                Document(
                    page_content=f"Paper title: {title}\nSection: {section}\nPage: {page_index + 1}\n\n{text}",
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "title": title,
                        "page": page_index,
                        "section": section,
                        "priority": priority,
                    },
                )
            )

        if overview_parts:
            documents.append(
                Document(
                    page_content=f"Paper title: {title}\nHigh-value paper overview:\n\n" + "\n\n".join(overview_parts[:8]),
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "title": title,
                        "page": 0,
                        "section": "paper overview",
                        "priority": 4.0,
                    },
                )
            )

    return documents, page_count


def list_pdfs():
    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    papers = []

    for pdf in sorted(PDF_DIRECTORY.glob("*.pdf")):
        title = pdf.stem
        try:
            reader = PdfReader(str(pdf))
            first_page_text = reader.pages[0].extract_text() if reader.pages else ""
            title = extract_title(pdf, first_page_text)
        except Exception:
            pass

        papers.append({"name": pdf.name, "title": title, "size": pdf.stat().st_size})

    return papers


def build_vectorstore():
    embeddings = get_embeddings()
    documents, page_count = extract_pdf_documents()

    if not documents:
        raise RuntimeError(f"No PDF files found in {PDF_DIRECTORY}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIRECTORY))
    return vectorstore, len(chunks), page_count


def load_or_build_vectorstore():
    embeddings = get_embeddings()
    index_file = VECTORSTORE_DIRECTORY / "index.faiss"

    if index_file.exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIRECTORY),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vectorstore, _, _ = build_vectorstore()
    return vectorstore


def query_intent_boost(question, section):
    lowered = question.lower()
    section = (section or "").lower()
    boost = 0.0
    intent_sections = {
        "summar": {"paper overview", "abstract", "introduction", "conclusion"},
        "overview": {"paper overview", "abstract", "introduction", "conclusion"},
        "method": {"methodology", "methods", "approach", "model", "experiments"},
        "architecture": {"model", "methodology", "approach"},
        "contribution": {"paper overview", "abstract", "introduction", "conclusion"},
        "finding": {"results", "discussion", "conclusion", "paper overview"},
        "limitation": {"limitations", "discussion", "conclusion", "future work"},
        "future": {"future work", "conclusion", "limitations"},
        "compare": {"paper overview", "abstract", "introduction", "conclusion"},
        "literature": {"paper overview", "abstract", "introduction", "conclusion", "limitations"},
    }

    for keyword, sections in intent_sections.items():
        if keyword in lowered and section in sections:
            boost += 0.9

    return boost


def title_match_boost(question, title):
    lowered = question.lower()
    title = (title or "").lower()
    boost = 0.0

    aliases = {
        "attention is all you need": {"transformer", "attention is all you need"},
        "language models are few-shot learners": {"gpt-3", "gpt3", "few-shot", "few shot"},
        "robust speech recognition via large-scale weak supervision": {"whisper", "speech recognition"},
        "deep residual learning for image recognition": {"resnet", "residual", "image recognition"},
    }

    for known_title, terms in aliases.items():
        if known_title in title and any(term in lowered for term in terms):
            boost += 1.4

    title_words = {
        word
        for word in re.findall(r"[a-z0-9]+", title)
        if len(word) > 3 and word not in {"paper", "learning", "models"}
    }
    if title_words:
        matched = sum(1 for word in title_words if word in lowered)
        if matched >= 2:
            boost += 1.0

    return boost


def retrieve_relevant_docs(vectorstore, question, limit=7):
    scored_docs = vectorstore.similarity_search_with_score(question, k=24)
    reranked = []

    for doc, distance in scored_docs:
        priority = float(doc.metadata.get("priority", 1.0))
        section = doc.metadata.get("section", "body")
        title = doc.metadata.get("title", "")
        boosted_score = (
            float(distance)
            - (priority * 0.18)
            - query_intent_boost(question, section)
            - title_match_boost(question, title)
        )
        reranked.append((boosted_score, doc))

    reranked.sort(key=lambda item: item[0])
    selected = []
    seen = set()

    for _, doc in reranked:
        key = (
            doc.metadata.get("filename"),
            doc.metadata.get("page"),
            doc.metadata.get("section"),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(doc)
        if len(selected) >= limit:
            break

    return selected


def get_granite_model():
    if not IBM_CLOUD_API_KEY or IBM_CLOUD_API_KEY == "your_ibm_cloud_api_key_here":
        raise RuntimeError("Set IBM_CLOUD_API_KEY in .env before asking questions.")

    credentials = Credentials(url=WATSONX_URL, api_key=IBM_CLOUD_API_KEY)
    return ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={
            "max_tokens": 700,
            "temperature": 0.2,
        },
    )


def build_prompt(question, docs):
    context = "\n\n".join(
        f"Paper: {doc.metadata.get('title', Path(doc.metadata.get('source', 'Unknown')).stem)}\n"
        f"Source file: {Path(doc.metadata.get('source', 'Unknown')).name}\n"
        f"Page: {int(doc.metadata.get('page', 0)) + 1}\n"
        f"Section: {doc.metadata.get('section', 'unknown')}\n"
        f"Context:\n{doc.page_content}"
        for doc in docs
    )

    return f"""Research context:
{context}

Question:
{question}

Answer:"""


def system_prompt():
    return """You are ResearchPilot, a careful research assistant for academic papers.
Use only the retrieved context supplied by the application.
Do not invent paper titles, acronyms, methods, datasets, metrics, or claims.
If a detail is not present in the context, say that it is not available in the retrieved excerpts.
Infer the user's task naturally from the question: summary, methodology, comparison, literature review, findings, limitations, future work, or direct Q&A.
Give a direct answer first. When appropriate, use short headings such as Objective, Methodology, Key Findings, Limitations, Future Work, and Conclusion.
Cite paper titles and page numbers for specific claims."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/papers", methods=["GET"])
def papers():
    return jsonify({"papers": list_pdfs()})


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("papers")

    if not files:
        return jsonify({"error": "Upload at least one PDF file."}), 400

    PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved = []

    for file in files:
        if not file or not file.filename:
            continue

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": f"{file.filename} is not a PDF file."}), 400

        filename = secure_filename(file.filename)
        file.save(PDF_DIRECTORY / filename)
        saved.append(filename)

    if not saved:
        return jsonify({"error": "No valid PDF files were uploaded."}), 400

    try:
        _, chunk_count, page_count = build_vectorstore()
    except Exception as exc:
        return jsonify({"error": f"Files uploaded, but indexing failed: {exc}"}), 500

    return jsonify(
        {
            "message": f"Uploaded {len(saved)} PDF file(s) and indexed {chunk_count} chunks from {page_count} pages.",
            "uploaded": saved,
            "papers": list_pdfs(),
        }
    )


@app.route("/api/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a research question."}), 400

    try:
        vectorstore = load_or_build_vectorstore()
        docs = retrieve_relevant_docs(vectorstore, question)
        prompt = build_prompt(question, docs)
        model = get_granite_model()
        response = model.chat(
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )
        answer = response["choices"][0]["message"]["content"]

        sources = sorted(
            {
                doc.metadata.get("title") or Path(doc.metadata.get("source", "Unknown")).name
                for doc in docs
                if doc.metadata.get("source")
            }
        )

        evidence = [
            {
                "title": doc.metadata.get("title") or Path(doc.metadata.get("source", "Unknown")).stem,
                "source": Path(doc.metadata.get("source", "Unknown")).name,
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section", "unknown"),
            }
            for doc in docs
        ]

        return jsonify({"answer": answer, "sources": sources, "evidence": evidence})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reindex", methods=["POST"])
def reindex():
    try:
        _, chunk_count, page_count = build_vectorstore()
        return jsonify({"message": f"Indexed {chunk_count} chunks from {page_count} pages.", "papers": list_pdfs()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
