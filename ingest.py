import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", message=".*fontTools.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
TEXTBOOKS_DIR = ROOT / "textbooks"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "unt_geography_v2"
EMBEDDING_MODEL = "gemini-embedding-001"
# Small batches keep a 429 from discarding a long run. Each batch is stored
# before the next one starts.
BATCH_SIZE = 16
TARGET_CHARS = 850
HARD_CHARS = 1200
MIN_CHARS = 80
CHUNK_REV = 2
CACHE_DIR = Path("/tmp/unt_v2_cache")

load_dotenv(ROOT / ".env")

_HEADING = re.compile(
    r"^(?:"
    r"§\s*\d+"
    r"|\d+(?:\.\d+){1,3}\.?\s+\S"
    r"|\d+\s*[-–—]\s*(?:тарау|сарамандық|зертханалық)\b"
    r"|[А-ЯӘҒҚҢӨҰҮҺІA-Z][А-ЯӘҒҚҢӨҰҮҺІA-Z0-9 «»\"'()\-]{7,}$"
    r")",
    re.IGNORECASE,
)
_WATERMARK = (
    "OKULYK",
    "Все учебники",
    "Книга предоставлена",
    "исключительно в образовательных",
    "Приказа Министра",
    "okulyk.com",
    "okulyk.kz",
)


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def grade_of(path: Path) -> int:
    match = re.match(r"\s*(\d{1,2})", path.name)
    if not match:
        print(f"Сынып нөмірі табылмады: {path.name}")
        sys.exit(1)
    grade = int(match.group(1))
    if grade not in range(7, 12):
        print(f"Сынып 7–11 аралығында емес: {path.name}")
        sys.exit(1)
    return grade


def ordered_pdfs() -> list[Path]:
    if not TEXTBOOKS_DIR.is_dir():
        return []
    pdfs = [
        path
        for path in TEXTBOOKS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(pdfs, key=lambda path: (grade_of(path), path.name))


def _collapse(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _is_watermark(line: str) -> bool:
    folded = line.lower()
    return any(mark.lower() in folded for mark in _WATERMARK)


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", line))


def _is_heading(line: str) -> bool:
    if len(line) > 90:
        return False
    if _HEADING.match(line):
        return True
    if line.endswith(":") and len(line) <= 70:
        return True
    return False


def _clean_page(text: str) -> str:
    kept: list[str] = []
    for raw in text.splitlines():
        if _is_watermark(raw):
            continue
        kept.append(raw)
    return "\n".join(kept)


def _usable_page(text: str) -> bool:
    if text.count("/uni") >= 4:
        return False
    body = _clean_page(text).strip()
    if len(body) < MIN_CHARS:
        return False
    letters = re.sub(r"[^0-9A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё]", "", body)
    return len(letters) >= 40


def _blocks(text: str) -> list[tuple[str, str]]:
    """Return (kind, text) where kind is 'heading' or 'body'.

    Full justified lines stay in the same paragraph. A short line ends it,
    and a heading line starts a new section.
    """
    rows: list[tuple[int, str]] = []
    for raw in _clean_page(text).splitlines():
        collapsed = _collapse(raw)
        if not collapsed or _is_page_number(collapsed):
            continue
        rows.append((len(raw.strip()), collapsed))
    blocks: list[tuple[str, str]] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        piece = buf.strip()
        buf = ""
        if piece:
            blocks.append(("body", piece))

    for index, (raw_len, line) in enumerate(rows):
        nxt = rows[index + 1][1] if index + 1 < len(rows) else ""
        wrapped = bool(nxt) and nxt[:1].islower() and raw_len < 58
        if _is_heading(line) and not wrapped and not line.endswith("-"):
            flush()
            blocks.append(("heading", line))
            continue
        if not buf:
            buf = line
        elif buf.endswith("-"):
            buf = buf[:-1] + line
        else:
            buf += " " + line
        # A full typeset line is long before the extra justification spaces
        # are collapsed. A short raw line is the end of a paragraph.
        if raw_len < 58 and not line.endswith("-"):
            if not (nxt and nxt[:1].islower() and not re.search(r"[.!?…]$", line)):
                flush()
    flush()
    return blocks


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _pack(paragraphs: list[str]) -> list[str]:
    """Pack paragraphs. Overlap one sentence only when a paragraph is split."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    def emit(parts: list[str]) -> None:
        text = "\n".join(part for part in parts if part).strip()
        if len(text) >= MIN_CHARS:
            pieces.append(text)

    for paragraph in paragraphs:
        if len(paragraph) <= HARD_CHARS:
            if current and size + len(paragraph) > TARGET_CHARS:
                emit(current)
                current = []
                size = 0
            current.append(paragraph)
            size += len(paragraph) + 1
            continue
        if current:
            emit(current)
            current = []
            size = 0
        overlap = ""
        group: list[str] = []
        group_size = 0
        for sentence in _sentences(paragraph):
            if group and group_size + len(sentence) > TARGET_CHARS:
                emit(group)
                group = [overlap] if overlap else []
                group_size = len(overlap)
            group.append(sentence)
            group_size += len(sentence) + 1
            overlap = sentence
        if group:
            current = [" ".join(group)]
            size = len(current[0])
    if current:
        emit(current)
    return pieces


def _page_chunks(path: Path, grade: int, page_no: int, text: str, heading: str) -> tuple[list[Document], str]:
    body_parts: list[str] = []
    docs: list[Document] = []
    section = heading
    ordinal = 0
    file_key = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]

    def take(paragraphs: list[str], section_name: str) -> None:
        nonlocal ordinal
        for piece in _pack(paragraphs):
            ordinal += 1
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": path.name,
                        "page": page_no,
                        "grade": grade,
                        "heading": section_name[:180],
                        "chunk_id": f"g{grade}_{file_key}_p{page_no:04d}_c{ordinal:02d}",
                    },
                )
            )

    for kind, piece in _blocks(text):
        if kind == "heading":
            if body_parts:
                take(body_parts, section)
                body_parts = []
            section = piece
            body_parts.append(piece)
            continue
        body_parts.append(piece)
    if body_parts:
        take(body_parts, section)
    return docs, section


def _ocr_png(png_path: str) -> str:
    # One thread per process. OpenMP oversubscribe makes a page take minutes.
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    try:
        proc = subprocess.run(
            ["tesseract", png_path, "stdout", "-l", "kaz+rus", "--psm", "6", "--oem", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=40,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout or ""


def _page_ranges(pages: list[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(pages))
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    return ranges


def _render_ocr(path: Path, page_numbers: list[int]) -> dict[int, str]:
    """OCR 1-based page numbers. Scanned pages have no text layer."""
    if not page_numbers:
        return {}
    out: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="unt-ocr-") as tmp:
        folder = Path(tmp)
        print(f"  сурет салынуда: {len(page_numbers)} бет", flush=True)
        for start, end in _page_ranges(page_numbers):
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(start),
                    "-l",
                    str(end),
                    "-r",
                    "150",
                    "-png",
                    str(path),
                    str(folder / "p"),
                ],
                check=True,
                capture_output=True,
            )
        jobs: list[tuple[int, str]] = []
        for image in sorted(folder.glob("p-*.png")):
            match = re.search(r"-(\d+)\.png$", image.name)
            if not match:
                continue
            page_no = int(match.group(1))
            if page_no in set(page_numbers):
                jobs.append((page_no, str(image)))
        if not jobs:
            return {}
        workers = min(4, os.cpu_count() or 2)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for page_no, text in zip(
                (page for page, _image in jobs),
                pool.map(_ocr_png, [image for _page, image in jobs]),
            ):
                out[page_no] = text
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  OCR {done}/{len(jobs)}", flush=True)
    return out


def _cache_file(path: Path) -> Path:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]
    return CACHE_DIR / f"{digest}.jsonl"


def _read_cache(path: Path) -> list[Document] | None:
    cache = _cache_file(path)
    if not cache.is_file():
        return None
    documents: list[Document] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rev") != CHUNK_REV:
            return None
        documents.append(Document(page_content=row["text"], metadata=row["meta"]))
    return documents


def _write_cache(path: Path, documents: list[Document]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_file(path)
    temporary = cache.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(
                json.dumps(
                    {"rev": CHUNK_REV, "text": doc.page_content, "meta": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(cache)


def load_chunks(pdfs: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in pdfs:
        grade = grade_of(path)
        cached = _read_cache(path)
        if cached is not None:
            print(f"Кэш: {grade}-сынып {path.name} — бөлік: {len(cached)}", flush=True)
            documents.extend(cached)
            continue
        print(f"Оқылуда: {grade}-сынып {path.name}", flush=True)
        try:
            reader = PdfReader(str(path))
        except Exception:
            print(f"PDF оқылмады: {path.name}. Файлды тексеріп, қайта көріңіз.")
            sys.exit(1)
        extracted = [(page.extract_text() or "") for page in reader.pages]
        weak = [index + 1 for index, text in enumerate(extracted) if not _usable_page(text)]
        ocr_text: dict[int, str] = {}
        if weak:
            print(f"  сурет беті: {len(weak)}", flush=True)
            ocr_text = _render_ocr(path, weak)
        heading = ""
        kept_pages = 0
        file_docs: list[Document] = []
        for index, raw in enumerate(extracted):
            page_no = index + 1
            if not _usable_page(raw):
                raw = ocr_text.get(page_no, "")
            if not _usable_page(raw):
                continue
            kept_pages += 1
            page_docs, heading = _page_chunks(path, grade, page_no, raw, heading)
            file_docs.extend(page_docs)
        _write_cache(path, file_docs)
        documents.extend(file_docs)
        print(f"  бет: {kept_pages}, бөлік: {len(file_docs)}", flush=True)
    return documents


def _gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    secret = _gemini_api_key()
    if secret:
        text = text.replace(secret, "***")
    return text[:240]


def _embed_batch(embeddings: GoogleGenerativeAIEmbeddings, texts: list[str]) -> list[list[float]]:
    delay = 2.0
    for attempt in range(8):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            message = _safe_error(exc)
            retry = "429" in message or "RESOURCE_EXHAUSTED" in message or "rate" in message.lower()
            if not retry or attempt == 7:
                raise
            print(f"Кідіріс {delay:.0f}с, сұраныс қайталанады.", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 90.0)
    return []


def _existing_ids(collection) -> set[str]:
    found: set[str] = set()
    offset = 0
    while True:
        batch = collection.get(limit=1000, offset=offset, include=[])
        ids = batch.get("ids") or []
        if not ids:
            break
        found.update(ids)
        offset += len(ids)
        if len(ids) < 1000:
            break
    return found


def store_chunks(chunks: list[Document], api_key: str) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # The live unt_geography collection stays until the new index is checked.
    if COLLECTION_NAME in _collection_names(client):
        collection = client.get_collection(COLLECTION_NAME)
    else:
        collection = client.create_collection(COLLECTION_NAME)
    done = _existing_ids(collection)
    pending = [doc for doc in chunks if doc.metadata["chunk_id"] not in done]
    print(f"Дайын бөлік: {len(done)}. Қалғаны: {len(pending)}.")
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        vectors = _embed_batch(embeddings, [doc.page_content for doc in batch])
        collection.add(
            ids=[doc.metadata["chunk_id"] for doc in batch],
            documents=[doc.page_content for doc in batch],
            metadatas=[
                {
                    "source": doc.metadata["source"],
                    "page": int(doc.metadata["page"]),
                    "grade": int(doc.metadata["grade"]),
                    "heading": doc.metadata.get("heading") or "",
                }
                for doc in batch
            ],
            embeddings=vectors,
        )
        saved = len(done) + min(start + len(batch), len(pending))
        print(f"Сақталды: {saved}/{len(chunks)}", flush=True)
        time.sleep(1.0)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    api_key = _gemini_api_key()
    if not api_key and not dry_run:
        print("GEMINI_API_KEY немесе GOOGLE_API_KEY табылмады. .env файлына кілтті қосыңыз.")
        sys.exit(1)

    pdfs = ordered_pdfs()
    if not pdfs:
        print(
            "textbooks/ қалтасы жоқ немесе онда PDF файл жоқ. "
            "Оқулықтардың PDF файлдарын textbooks/ қалтасына салыңыз."
        )
        sys.exit(1)

    print("Сынып реті:")
    for path in pdfs:
        print(f"  {grade_of(path)}  {path.name}")

    total = 0
    failed: list[str] = []
    for path in pdfs:
        try:
            chunks = load_chunks([path])
        except Exception as exc:
            print(f"Қате (оқу): {path.name}: {_safe_error(exc)}", flush=True)
            failed.append(path.name)
            continue
        total += len(chunks)
        if dry_run or not chunks:
            continue
        try:
            store_chunks(chunks, api_key)
        except Exception as exc:
            print(f"Қате (вектор): {path.name}: {_safe_error(exc)}", flush=True)
            failed.append(path.name)
    print(f"PDF саны: {len(pdfs)}")
    print(f"Бөлік саны: {total}")
    if failed:
        print("Аяқталмаған файлдар: " + "; ".join(failed))
        sys.exit(1)
    if total == 0:
        print("PDF файлдарынан мәтін алынбады. Басқа оқулықты қолданып көріңіз.")
        sys.exit(1)
    if dry_run:
        return
    print(f"Жинақ: {COLLECTION_NAME}")
    print(f"Дерекқор сақталды: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
