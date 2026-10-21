import asyncio
import html
import json
import logging
import os
import random
import re
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import chromadb
import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "unt_geography_v2"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
TELEGRAM_LIMIT = 4096
TOP_K = 10
# Free RAG sends this many textbook chunks to the model.
RAG_TOP_K = 8
# Cosine floor. Partial wording and a synonym sit below a tight match, so a
# semantic neighbor stays in the pool. Exact keywords are not required.
SOFT_MIN_COSINE = 0.32
# High lambda keeps similar wording instead of swapping it for a distant page.
MMR_LAMBDA = 0.95
MMR_CANDIDATES = 40
FALLBACK_NOTE = "(💡 Жалпы географиялық білім қорынан)"
NOT_FOUND_FACT = "Бұл сұрақ бойынша нақты мәлімет табылмады."
TYPING_INTERVAL = 4.0
TOPIC_EMOJIS = ("🌍", "📚", "📌", "💡")
ENCOURAGEMENTS = (
    "Тамаша сұрақ! ҰБТ-да бұл сұрақ міндетті түрде кездеседі, сәттілік! 🚀",
    "Жаттығуды жалғастыр, сен бәрін жеңесің! 🔥",
    "Керемет! Осы тақырыпты тағы бір рет қайталасаң, ҰБТ-да сенімді боласың. ✨",
    "Жақсы қарқын! Әр сұрақ сені ҰБТ-ға жақындатады. 💪",
    "Дәл осындай сұрақтарды жиі шеш, нәтиже өседі. 🌟",
)

# The collection's HNSW graph returns the same OKULYK watermark pages for
# every query, and the distance it reports does not match the stored vectors.
# Those vectors are valid (they match a fresh gemini-embedding-001 embed), so
# ranking is done against the stored matrix instead of collection.query.
_ACRONYM = re.compile(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё]{3,8}")
_STYLE = (
    "Сен ҰБТ география пәнінің тәлімгерісің. Жауапты қазақ тілінде жаз.\n"
    "Пішін: бірінші жол осы жиынтықтан кемінде екі тақырып эмодзиімен басталсын: "
    "🌍 📚 📌 💡. Содан кейін бір тығыз абзац емес, әр деректі жеке жолда • тармағымен жаз.\n"
    "Мотивациялық сөйлем қоспа.\n"
    "Егер оқушы нақты кен орнын, қаланы, өзенді немесе нысанды сұраса "
    "(мысалы: Екібастұз алабы, Каспий өзендері), жалпы теорияны жазба. "
    "«бассейн дегеніміз не» немесе «кен орны деген не» сияқты түсіндірме жоқ. "
    "Тек сол сұраққа жауап беретін атауларды, кен орындарын немесе сандарды "
    "🔹 белгісімен таза тізім етіп жаз. Сұралған нысанды атамайтын толтырма мен "
    "оқулық теориясын таста.\n"
)
_SPECIFIC_ASK = re.compile(
    r"алаб|бассейн|кен\s*орн|өзендер|өзені|құятын|қалалар|салалар",
    re.IGNORECASE,
)
_OPEN_ASK = re.compile(r"(қайда|неге|қалай|неліктен|қандай|не\s+үшін)", re.IGNORECASE)
_THEORY = re.compile(
    r"дегеніміз|деп атала|деген не|анықтама|түсіндірме|теория|"
    r"бассейн деп|кен орны деген|тапсырма|материал",
    re.IGNORECASE,
)
_LIST_MARK = re.compile(r"^(?:[•*\-–—]|\d+[.)])\s*")
_GENERIC_ASK = {
    "алабы",
    "алап",
    "алабына",
    "өзендер",
    "өзен",
    "өзені",
    "өзендері",
    "құятын",
    "құяды",
    "бассейн",
    "бассейні",
    "кен",
    "орны",
    "қала",
    "қалалар",
    "теңіз",
    "теңізі",
    "теңізіне",
    "теңізге",
}
TEXTBOOK_PROMPT = (
    "Сен ҰБТ география тәлімгерісің. Жауапты қазақ тілінде, қысқа, анық, сауатты жаз.\n"
    "Үзіндіні көшірме және сынған сөз тізбесін шығарма. "
    "Үзінділер тақырып туралы нақты немесе ішінара мәлімет берсе, солардан екі-төрт сөйлем жауап құра. "
    "Жыл, атау және сабақтас фактіні жасырма.\n"
    "Аппенин елдері десе, тек сол түбектегі елді ата (Италия; үзіндіде болса Ватикан). "
    "АҚШ, Жапония және басқа елдерді қоспа.\n"
    "Каспийге құятын өзен десе, тек Каспийге құятыны айтылған өзенді немесе алапты ата. "
    "Басқа бассейндерді қоспа.\n"
    "Биік нүкте сұралса, шыңның атын түсінікті жаз, кестені көшірме.\n"
    "Қай жылы ашылғаны сұралса, құрылған, ұйымдастырылған және негізі қаланған — сол мағына. "
    "Үзіндідегі жылды жаз.\n"
    "Көмірдің түрі немесе сапасы сұралса, үзіндідегі антрацит, таскөмір, қоңыр көмір немесе кокс туралы сөйлемді жауапқа қос.\n"
    "«Бұл сұрақ бойынша нақты мәлімет табылмады» деп тек үзінділердің ешқайсысы тақырып туралы болмағанда ғана жаз. "
    "Жыл, атау немесе сабақтас факт тұрса, сол сөйлемді жазба.\n"
    "Дереккөз жолын және мотивация жазба. "
    "(💡 Жалпы географиялық білім қорынан) жолын қоспа."
)
SYNTHESIZE_PROMPT = (
    "Сен ҰБТ география тәлімгерісің. Берілген үзінділер осы сұраққа қатысты. "
    "Солардан қысқа, нақты қазақша жауап құра: екі-төрт сөйлем. "
    "Жыл, атау немесе сабақтас фактіні сол қалпында жаз. "
    "«Бұл сұрақ бойынша нақты мәлімет табылмады» деп жазба. "
    "Дереккөз жолын жазба."
)
REWRITE_PROMPT = (
    "Сен географиялық іздеу сұрағын нақтылайсың. "
    "Оқушы сұрағын оқулықтан іздеуге ыңғайлы бір жолға айналдыр: "
    "нақты атаулар мен кілт сөздер болсын (тау, шың, өзен, теңіз, биіктік). "
    "Мысал: «Алтай тауының биік нүктесі» → Алтай Мұзтау биіктігі. "
    "Тек іздеу жолын жаз. Түсіндірме, тырнақша және бірнеше нұсқа жоқ."
)
RERANK_PROMPT = (
    "Сұраққа тікелей жауап беретін үзінді нөмірлерін жаз. Ең көбі 4 нөмір, үтірмен. "
    "Жалпы анықтаманы және сұраққа жатпайтын ел, өзен, тауды алма. "
    "Ешқайсысы жауап бермесе, тек 0 жаз. Басқа мәтін жоқ."
)
_FOCUS_STOP = _GENERIC_ASK | {
    "биіктігі",
    "биіктік",
    "биік",
    "нүктесі",
    "нүкте",
    "тауының",
    "тауы",
    "таудың",
    "тау",
    "сағасы",
    "саға",
    "өзенінің",
    "өзеннің",
    "қайда",
    "құяды",
    "құятын",
    "жоғары",
    "шыңы",
    "шың",
}
FALLBACK_PROMPT = (
    _STYLE
    + "Оқулықтан үзінді табылмады. Сұраққа өз географиялық біліміңмен толық жауап бер. "
    "Файл аты мен бет нөмірін атама. "
    "Жауапты дәл мына жолмен бітір, бөлек жолда:\n"
    "(💡 Жалпы географиялық білім қорынан)"
)
MISSING_DB = "Дерекқор табылмады. Алдымен ingest.py скриптін іске қосыңыз."
API_ERROR = "Қате шықты. Сәл кейінірек қайталап көріңіз."
ONE_QUESTION_USED = "Сіздің бір сұрағыңыз қолданылып қойды."
ASKED_PATH = ROOT / "data" / "asked_users.json"

load_dotenv(ROOT / ".env")

router = Router()

_client: chromadb.ClientAPI | None = None
_embeddings: GoogleGenerativeAIEmbeddings | None = None
_llm: ChatGoogleGenerativeAI | None = None
_index_lock = threading.Lock()
_cheer_lock = threading.Lock()
_cheer_index = 0
_asked_lock = threading.Lock()
_asked_users: set[str] | None = None
_matrix: np.ndarray | None = None
_docs: list[str] = []
_metas: list[dict] = []
_last_search_query = ""


def _require_env() -> tuple[str, str]:
    token = (
        os.getenv("BOT_TOKEN_1")
        or os.getenv("BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not token or not api_key:
        print(
            "BOT_TOKEN_1 (немесе BOT_TOKEN, TELEGRAM_BOT_TOKEN) және "
            "GEMINI_API_KEY (немесе GOOGLE_API_KEY) жоқ. .env файлын тексеріңіз."
        )
        sys.exit(1)
    return token, api_key


def _owner_ids() -> set[str]:
    raw = os.getenv("TELEGRAM_OWNER_ID", "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _is_owner_id(user_id: object) -> bool:
    """8091822446 has no free-pool, quiz, or daily cap. Match int or str."""
    key = str(user_id).strip() if user_id is not None else ""
    if not key:
        return False
    owners = set(_owner_ids())
    owners.add(ADMIN_ID)
    if key in owners:
        return True
    try:
        return str(int(key)) in owners
    except ValueError:
        return False


def _load_asked() -> set[str]:
    global _asked_users
    if _asked_users is not None:
        return _asked_users
    found: set[str] = set()
    if ASKED_PATH.is_file():
        try:
            data = json.loads(ASKED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = []
        if isinstance(data, list):
            found = {str(item) for item in data}
    _asked_users = found
    return found


def _save_asked(ids: set[str]) -> None:
    ASKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ASKED_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(sorted(ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ASKED_PATH)


def _claim_question(user_id: int | None) -> bool:
    """First question of a non-owner is stored. A later one is refused."""
    if user_id is None:
        return True
    if _is_owner_id(user_id):
        return True
    key = str(user_id)
    with _asked_lock:
        asked = _load_asked()
        if key in asked:
            return False
        asked.add(key)
        try:
            _save_asked(asked)
        except OSError:
            asked.discard(key)
            raise
        return True


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    # chromadb 0.6+ returns names; older releases return collection objects.
    names: set[str] = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def _open_client() -> chromadb.ClientAPI | None:
    global _client
    if not (CHROMA_DIR / "chroma.sqlite3").is_file():
        return None
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME not in _collection_names(_client):
        return None
    return _client


def _usable(text: str) -> bool:
    body = text.strip()
    if len(body) < 80:
        return False
    # Title pages that are only the OKULYK watermark. Real pages that also
    # carry the footer are longer than this.
    if "OKULYK" in body and len(body) < 280:
        return False
    # PDF extracts that stored glyph ids as "/uniXXXX" instead of letters.
    if body.count("/uni") >= 4:
        return False
    return True


# Each group adds search vectors. A hit never removes a chunk from the pool.
_SYNONYM_GROUPS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"ашыл\w*|құрыл\w*|ұйымдастыр\w*|негіз\w*\s+қалан\w*",
            re.IGNORECASE,
        ),
        ("ашылған", "құрылған", "ұйымдастырылған", "негізі қаланған"),
    ),
    (
        re.compile(r"полиметалл\w*|полиметал\w*", re.IGNORECASE),
        ("полиметалл", "полиметалл кені", "көп металды кен"),
    ),
    (
        re.compile(
            r"көмір\w*|антрацит\w*|антрашит\w*|таскөмір\w*|қоңыр\s*көмір\w*|кокс\w*",
            re.IGNORECASE,
        ),
        ("антрацит", "таскөмір", "қоңыр көмір", "кокстелетін көмір"),
    ),
)


def _query_forms(question: str) -> list[str]:
    """Extra wordings for the same question. They only add vectors to compare."""
    text = " ".join(question.split())
    forms = [text]
    words = text.split()
    for token in _ACRONYM.findall(text):
        # БРИКС and «брикс» share a phrase form with «Брикс елдері».
        if token.isupper() or (token.islower() and len(words) == 1):
            titled = token[0].upper() + token[1:].lower()
            forms.append(f"{titled} елдері")
    folded = text.lower()
    for pattern, synonyms in _SYNONYM_GROUPS:
        if not pattern.search(text):
            continue
        for synonym in synonyms:
            if synonym.lower() in folded:
                continue
            replaced = pattern.sub(synonym, text, count=1)
            forms.append(replaced if replaced != text else f"{text} {synonym}")
            forms.append(f"{text} {synonym}")
    unique: list[str] = []
    seen: set[str] = set()
    for form in forms:
        if form in seen:
            continue
        seen.add(form)
        unique.append(form)
    return unique


def _load_index() -> bool:
    global _matrix, _docs, _metas
    if _matrix is not None:
        return True
    client = _open_client()
    if client is None:
        return False
    collection = client.get_collection(COLLECTION_NAME)
    docs: list[str] = []
    metas: list[dict] = []
    vectors: list = []
    offset = 0
    while True:
        batch = collection.get(
            limit=1000,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        for doc, meta, vector in zip(
            batch["documents"], batch["metadatas"], batch["embeddings"]
        ):
            text = doc or ""
            if not _usable(text):
                continue
            docs.append(text)
            metas.append(meta or {})
            vectors.append(vector)
        offset += len(ids)
        if len(ids) < 1000:
            break
    if not vectors:
        return False
    _docs = docs
    _metas = metas
    _matrix = np.asarray(vectors, dtype=np.float32)
    print(f"Үзінді индексі: {len(docs)}")
    return True


def _ensure_index() -> bool:
    if _matrix is not None:
        return True
    with _index_lock:
        return _load_index()


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    return _embeddings


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key)
    return _llm


def _cosine(vectors: np.ndarray) -> np.ndarray:
    """Max cosine of each stored chunk against every query vector."""
    assert _matrix is not None
    row_norms = np.linalg.norm(_matrix, axis=1)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    query_norms = np.linalg.norm(vectors, axis=1)
    query_norms = np.where(query_norms == 0, 1.0, query_norms)
    scores = (_matrix @ vectors.T) / (row_norms[:, None] * query_norms[None, :])
    return scores.max(axis=1)


def _mmr_order(scores: np.ndarray, k: int) -> list[int]:
    """Pick k chunks: relevant to the question and not copies of each other."""
    assert _matrix is not None
    candidates: list[int] = []
    seen: set[str] = set()
    for index in np.argsort(-scores):
        if float(scores[int(index)]) < SOFT_MIN_COSINE:
            break
        text = _docs[int(index)].strip()
        if text in seen:
            continue
        seen.add(text)
        candidates.append(int(index))
        if len(candidates) >= MMR_CANDIDATES:
            break
    if not candidates:
        return []
    chosen_local: list[int] = [0]
    remaining = set(range(1, len(candidates)))
    vectors = _matrix[np.asarray(candidates, dtype=np.int64)]
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    units = vectors / norms[:, None]
    while remaining and len(chosen_local) < k:
        best_local = -1
        best_value = -1e9
        for local in remaining:
            redundancy = max(
                float(units[local] @ units[picked]) for picked in chosen_local
            )
            value = MMR_LAMBDA * float(scores[candidates[local]]) - (
                1.0 - MMR_LAMBDA
            ) * redundancy
            if value > best_value:
                best_value = value
                best_local = local
        chosen_local.append(best_local)
        remaining.remove(best_local)
    return [candidates[local] for local in chosen_local]


def _query_scores(question: str, api_key: str, also: str = "") -> np.ndarray | None:
    if not _ensure_index() or _matrix is None:
        return None
    embeddings = get_embeddings(api_key)
    forms = _query_forms(question)
    if also and also != question:
        for form in _query_forms(also):
            if form not in forms:
                forms.append(form)
    # Chunks were embedded as documents. A document-side vector of the question
    # sits in that space; the retrieval-query vector covers a different wording.
    vectors = [np.asarray(embeddings.embed_query(question), dtype=np.float32)]
    vectors.extend(
        np.asarray(row, dtype=np.float32) for row in embeddings.embed_documents(forms)
    )
    return _cosine(np.stack(vectors))


def _focus_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(
        r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", query.lower()
    ):
        if word in _FOCUS_STOP or word in seen:
            continue
        seen.add(word)
        terms.append(word)
    return terms


def _term_hits(text: str, terms: list[str]) -> int:
    if not terms:
        return 0
    low = text.lower()
    return sum(1 for term in terms if term in low)


def retrieve(
    question: str, api_key: str, k: int = TOP_K
) -> list[tuple[Document, float]] | None:
    scores = _query_scores(question, api_key)
    if scores is None:
        return None
    hits: list[tuple[Document, float]] = []
    for index in _mmr_order(scores, k):
        hits.append(
            (
                Document(page_content=_docs[index], metadata=_metas[index]),
                float(scores[index]),
            )
        )
    return hits


def candidate_pool(
    question: str, api_key: str, k: int = TOP_K, also: str = ""
) -> list[tuple[Document, float]] | None:
    """Semantic neighbors. A keyword or a synonym never drops one of them."""
    scores = _query_scores(question, api_key, also)
    if scores is None:
        return None
    return [
        (
            Document(page_content=_docs[index], metadata=_metas[index]),
            float(scores[index]),
        )
        for index in _mmr_order(scores, k)
    ]


def _context_block(docs: list) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        label = []
        if source:
            label.append(f"файл: {source}")
        if page is not None:
            label.append(f"бет: {page}")
        header = f"[{', '.join(label)}]" if label else "[дереккөз жоқ]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _as_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def _citation(doc: Document) -> str:
    source = str(doc.metadata.get("source") or "белгісіз")
    page = doc.metadata.get("page")
    if page is None:
        return f"[Дереккөз: {source}]"
    return f"[Дереккөз: {source}, {page}]"


def _ensure_citations(body: str, docs: list) -> str:
    body = re.sub(r"\s*\[Дереккөз:[^\]]*\]\.?", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    cites: list[str] = []
    seen: set[str] = set()
    for doc in docs[:4]:
        cite = _citation(doc)
        if cite in seen:
            continue
        seen.add(cite)
        cites.append(cite)
    if not cites:
        return body
    return body + "\n" + "\n".join(cites)


def _next_encouragement() -> str:
    global _cheer_index
    with _cheer_lock:
        line = ENCOURAGEMENTS[_cheer_index % len(ENCOURAGEMENTS)]
        _cheer_index += 1
        return line


def _specific_object_question(question: str) -> bool:
    if _OPEN_ASK.search(question) and not _SPECIFIC_ASK.search(question):
        return False
    if _SPECIFIC_ASK.search(question):
        return True
    words = [word for word in re.split(r"\s+", question.strip(" ?!.")) if word]
    return 1 <= len(words) <= 4


def _clean_fact(line: str) -> str:
    line = _LIST_MARK.sub("", line.strip())
    line = re.sub(r"[🔹🌍📚📌💡]", "", line)
    line = line.strip(" \t-–—•")
    line = re.sub(r"\s*\[Дереккөз:[^\]]*\]\.?", "", line).strip()
    return line


def _is_filler(line: str) -> bool:
    letters = re.sub(r"[^0-9A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]", "", line)
    if len(letters) < 2 or _THEORY.search(line):
        return True
    if len(line.split()) > 6 and not re.search(r"\d", line):
        return True
    return False


def _facts_from_text(text: str) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = _clean_fact(raw)
        if _is_filler(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(line)
    return facts


def _query_stems(question: str) -> set[str]:
    stems: set[str] = set()
    for word in re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", question.lower()):
        if word in _GENERIC_ASK:
            continue
        stems.add(word if len(word) <= 5 else word[:5])
    return stems


def _names_near_question(question: str, docs: list) -> list[str]:
    stems = _query_stems(question)
    if not stems:
        return []
    found: list[str] = []
    seen: set[str] = set()
    name = re.compile(
        r"[A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,}"
        r"(?:\s*\([^)\n]{1,24}\))?"
    )
    for doc in docs:
        text = doc.page_content.replace("\n", " ")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            low = sentence.lower()
            if not any(stem in low for stem in stems):
                continue
            if _THEORY.search(sentence):
                continue
            for match in name.findall(sentence):
                item = match.strip(" -–—")
                key = item.lower()
                if key in seen or key in _GENERIC_ASK:
                    continue
                seen.add(key)
                found.append(item)
    return found[:12]


_GENERIC_DEPOSIT = {
    "қазба",
    "қазбалар",
    "пайдалы",
    "кен",
    "кендер",
    "көмір",
    "мұнай",
    "газ",
    "темір",
    "шикізат",
    "отын",
    "руда",
    "рудалар",
}
_BANNED_NAMES = {
    "әртүрлі",
    "келесі",
    "геграфтар",
    "географтар",
    "географ",
    "география",
    "географиялық",
    "дереккөз",
    "дереккөздер",
    "мысалы",
    "барлық",
    "басқа",
    "олар",
    "олардың",
    "бұл",
    "сол",
    "мына",
    "әрі",
    "әр",
    "сурет",
    "тапсырма",
    "мәтін",
    "параграф",
    "материал",
    "қарастыр",
    "анықта",
    "сипатта",
    "оңтүстік",
    "солтүстік",
    "батыс",
    "шығыс",
    "орталық",
    "қазақстан",
    "қазақстанның",
    "республика",
    "облыс",
    "облысы",
    "аудан",
    "ауданы",
}
_COUNTRIES = (
    "Сан-Марино",
    "Біріккен Араб Әмірліктері",
    "Сауд Арабиясы",
    "Оңтүстік Корея",
    "Солтүстік Корея",
    "Жаңа Зеландия",
    "Ұлыбритания",
    "Шри-Ланка",
    "Коста-Рика",
    "Ватикан",
    "Италия",
    "Франция",
    "Германия",
    "Испания",
    "Португалия",
    "Грекия",
    "Түркия",
    "Ресей",
    "Қытай",
    "Үндістан",
    "Жапония",
    "Аустралия",
    "Канада",
    "Бразилия",
    "Аргентина",
    "Мексика",
    "Мысыр",
    "Египет",
    "Польша",
    "Чехия",
    "Словакия",
    "Венгрия",
    "Румыния",
    "Болгария",
    "Сербия",
    "Хорватия",
    "Словения",
    "Албания",
    "Украина",
    "Беларусь",
    "Молдова",
    "Литва",
    "Латвия",
    "Эстония",
    "Финляндия",
    "Швеция",
    "Норвегия",
    "Дания",
    "Исландия",
    "Ирландия",
    "Нидерланд",
    "Бельгия",
    "Люксембург",
    "Швейцария",
    "Австрия",
    "Монако",
    "Андорра",
    "Мальта",
    "Кипр",
    "Грузия",
    "Армения",
    "Әзербайжан",
    "Иран",
    "Ирак",
    "Сирия",
    "Ливан",
    "Израиль",
    "Иордания",
    "Моңғолия",
    "Пәкістан",
    "Ауғанстан",
    "Өзбекстан",
    "Қырғызстан",
    "Түрікменстан",
    "Тәжікстан",
    "Қазақстан",
    "АҚШ",
    "ГФР",
)
_COUNTRY_SUFFIX = (
    r"(?:ның|нің|дың|дің|тың|тің|ға|ге|қа|ке|на|не|да|де|та|те|"
    r"дан|ден|тан|тен|мен|пен|бен)?"
)
_KIND_WORDS = _GENERIC_ASK | {
    "елдер",
    "елдері",
    "ел",
    "мемлекет",
    "мемлекеттер",
    "мемлекеттері",
    "түбек",
    "түбегі",
    "түбегінде",
    "түбегіндегі",
    "түбектері",
    "құятын",
    "құяды",
    "нысан",
    "нысандар",
    "нысандары",
    "қалалар",
    "қалалары",
    "таулар",
    "көлдер",
    "аралдар",
}


def _dehyphen(text: str) -> str:
    flat = text.replace("\r", "\n")
    flat = re.sub(r"-\s*\n\s*", "", flat)
    flat = re.sub(r"\s+", " ", flat)
    return flat.strip()


def _list_kind(question: str) -> str | None:
    folded = question.lower()
    if re.search(r"елдер|елдері|мемлекет", folded):
        return "country"
    if re.search(r"алаб|бассейн|кен\s*орн|кеніш", folded):
        return "deposit"
    if re.search(r"өзен", folded):
        return "river"
    if re.search(r"қалалар|қалалары|таулар|көлдер|аралдар|нысандар", folded):
        return "object"
    return None


def _topic_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", question.lower()):
        if word in _KIND_WORDS or word in _FOCUS_STOP or word in seen:
            continue
        stem = word if len(word) <= 6 else word[:6]
        if stem in seen:
            continue
        seen.add(stem)
        terms.append(stem)
    return terms


def _near_topic(text: str, start: int, end: int, topics: list[str]) -> bool:
    if not topics:
        return True
    window = text[max(0, start - 400) : end + 700].lower()
    return any(topic in window for topic in topics)


def _banned_name(name: str) -> bool:
    parts = re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,}", name.lower())
    if not parts:
        return True
    if any(part in _BANNED_NAMES or part in _GENERIC_ASK or part in _KIND_WORDS for part in parts):
        if not any(part not in _BANNED_NAMES and part not in _GENERIC_ASK and part not in _KIND_WORDS for part in parts):
            return True
    head = parts[0]
    if head in _BANNED_NAMES or head in _KIND_WORDS:
        return True
    if re.search(r"(?:ңдар|іңдер|ыңдар|іңіз|ыңыз|ңдар)$", head):
        return True
    return False


def _keep_longest(names: list[str]) -> list[str]:
    kept: list[str] = []
    for name in names:
        key = name.lower().split()[0]
        replaced = False
        for index, old in enumerate(kept):
            old_key = old.lower().split()[0]
            if key == old_key or name.lower() in old.lower() or old.lower() in name.lower():
                if len(name) > len(old):
                    kept[index] = name
                replaced = True
                break
        if not replaced:
            kept.append(name)
    return kept


# Countries that actually belong to the place named in the question.
# A country is kept only when the chunk writes that name.
_REGION_MEMBERS = (
    (("аппени", "апенни"), ("Италия", "Ватикан", "Сан-Марино")),
)


def _members_for(topics: list[str]) -> list[str]:
    members: list[str] = []
    for keys, names in _REGION_MEMBERS:
        if any(topic.startswith(key) or key.startswith(topic) for topic in topics for key in keys):
            members.extend(names)
    return members


def _country_names(text: str, topics: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    ordered = sorted(_COUNTRIES, key=len, reverse=True)
    for country in ordered:
        pattern = rf"(?<![\w]){re.escape(country)}{_COUNTRY_SUFFIX}(?![\w])"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            tail = text[match.end() : match.end() + 16].lower()
            if re.match(r"\s*(?:түбе|теңіз|көлі|тауы|өзен|құрлық|қала|арал)", tail):
                continue
            key = country.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append((match.start(), country))
            break
    found.sort()
    members = {name.lower() for name in _members_for(topics)}
    if members:
        return [name for _start, name in found if name.lower() in members]
    return [
        name
        for start, name in found
        if _near_topic(text, start, start + len(name), topics)
    ]


_RIVER_TOKEN = (
    r"[A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,}"
    r"(?:-[A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,})*"
    r"(?:\s*\([A-ZА-ЯӘҒҚҢӨҰҮҺІ][^)]{1,24}\))?"
)


def _skip_river(name: str) -> bool:
    if _banned_name(name):
        return True
    low = name.lower()
    if re.search(r"көл|теңіз|тау|түбе|құрлық|грэс", low):
        return True
    return low in {"бастау", "өзен", "бассейн", "бассейні", "шегара"}


def _name_list(blob: str) -> list[str]:
    names: list[str] = []
    plain = re.sub(r"\([^)]*\)", " ", blob)
    for part in re.split(r",| және ", plain):
        match = re.match(rf"\s*({_RIVER_TOKEN})", part.strip())
        if not match:
            continue
        name = " ".join(match.group(1).split())
        if _skip_river(name):
            continue
        names.append(name)
    return names


def _river_answers(text: str, start: int, name: str, topics: list[str]) -> bool:
    """The chunk must tie this river to the asked water, not just list basins."""
    if any(topic in name.lower() for topic in topics):
        return True
    at = text.find(name, max(0, start - 40))
    if at < 0:
        at = start
    left = max(text.rfind(mark, 0, at) for mark in ".?!")
    right = len(text)
    for mark in ".?!":
        pos = text.find(mark, at + len(name))
        if pos != -1:
            right = min(right, pos)
    if text[left + 1 : right].count(",") >= 2:
        return False
    window = text[max(0, at - 700) : at + len(name) + 700].lower()
    if not any(topic in window for topic in topics):
        return False
    return "құя" in window or "құйы" in window


def _river_names(text: str, topics: list[str], question: str) -> list[str]:
    found: list[tuple[int, str]] = []

    def add(start: int, name: str) -> None:
        name = " ".join(name.split())
        if _skip_river(name):
            return
        found.append((start, name))

    explicit = (
        rf"({_RIVER_TOKEN})\s*[–—-]\s*[^.]{{0,80}}?өзен",
        rf"({_RIVER_TOKEN})\s+өзен",
    )
    for pattern in explicit:
        for match in re.finditer(pattern, text):
            add(match.start(1), match.group(1))
    for match in re.finditer(r"өзендерге\s+(.{0,160}?)\s+жатады", text):
        for name in _name_list(match.group(1)):
            add(match.start(1), name)
    for match in re.finditer(r":\s*((?:[^.]{0,240}))", text):
        sentence = text[max(0, match.start() - 180) : match.end()].lower()
        if "өзен" not in sentence and not any(topic in sentence for topic in topics):
            continue
        for name in _name_list(match.group(1)):
            add(match.start(1), name)
    if not found:
        return []
    if not topics:
        return _keep_longest([name for _start, name in sorted(found)])
    linked = [
        name
        for start, name in sorted(found)
        if _river_answers(text, start, name, topics)
    ]
    return _keep_longest(linked)


def _deposit_names(text: str, topics: list[str]) -> list[str]:
    pattern = (
        r"([A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,})"
        r"(?:\s+(?:көмір|мұнай|газ|темір|мыс|алтын|уран|фосфор))?"
        r"\s+(?:бассейні|бассейн|алабы|алап|кеніші|кеніш|кен\s+орны)"
    )
    found: list[str] = []
    for match in re.finditer(pattern, text):
        name = " ".join(match.group(0).split())
        head = match.group(1).lower()
        if head in _GENERIC_DEPOSIT or _banned_name(name):
            continue
        found.append(name)
    linked = [name for name in found if topics and any(topic in name.lower() for topic in topics)]
    if linked:
        return _keep_longest(linked)
    if not topics:
        return []
    places: list[str] = []
    for match in re.finditer(
        r"[A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{3,}", text
    ):
        word = match.group(0)
        low = word.lower()
        if not any(low.startswith(topic) or topic in low for topic in topics):
            continue
        if _banned_name(word) or low in _GENERIC_DEPOSIT:
            continue
        window = text[max(0, match.start() - 300) : match.end() + 300].lower()
        if not re.search(r"кен|көмір|алап|бассейн|мұнай|қазба", window):
            continue
        places.append(word)
    return _keep_longest(places)


def _object_names(text: str, topics: list[str], question: str) -> list[str]:
    folded = question.lower()
    if "қала" in folded:
        marker = r"қала"
    elif "көл" in folded:
        marker = r"көл"
    elif "тау" in folded:
        marker = r"тау"
    elif "арал" in folded:
        marker = r"арал"
    else:
        marker = r"нысан"
    pattern = (
        r"([A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,}"
        r"(?:\s+[A-ZА-ЯӘҒҚҢӨҰҮҺІ][A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі-]{2,})?)"
        rf"\s+{marker}"
    )
    found: list[str] = []
    for match in re.finditer(pattern, text):
        name = " ".join(match.group(1).split())
        if _banned_name(name):
            continue
        if not _near_topic(text, match.start(), match.end(), topics):
            continue
        found.append(name)
    return _keep_longest(found)


def _strict_names(question: str, docs: list, kind: str) -> list[str]:
    topics = _topic_terms(question)
    chunks = [_dehyphen(doc.page_content) for doc in docs]
    names: list[str] = []
    for chunk in chunks:
        if kind == "country":
            names.extend(_country_names(chunk, topics))
        elif kind == "deposit":
            names.extend(_deposit_names(chunk, topics))
        elif kind == "river":
            names.extend(_river_names(chunk, topics, question))
        else:
            names.extend(_object_names(chunk, topics, question))
    return _keep_longest(names)[:12]


def _styled_list(docs: list, names: list[str]) -> str:
    if not names:
        return f"🌍 📚\n{NOT_FOUND_FACT}"
    body = "🌍 📚\n" + "\n".join(f"🔹 {name}" for name in names)
    body = _ensure_citations(body, docs)
    return f"{body}\n\n{_next_encouragement()}"


def _apply_precision(question: str, body: str, docs: list) -> str:
    if not _specific_object_question(question):
        return body
    facts = _facts_from_text(body)
    if not facts:
        facts = _names_near_question(question, docs)
    if not facts:
        return body
    return "\n".join(f"🔹 {fact}" for fact in facts)


def _styled_answer(answer: str, docs: list, *, fallback: bool, question: str) -> str:
    kept = [
        line
        for line in answer.splitlines()
        if line.strip() != FALLBACK_NOTE and line.strip() not in ENCOURAGEMENTS
    ]
    body = _apply_precision(question, "\n".join(kept).strip(), docs)
    first = body.splitlines()[0] if body else ""
    if not any(emoji in first for emoji in TOPIC_EMOJIS):
        body = f"🌍 📚\n{body}"
    cheer = _next_encouragement()
    if fallback:
        return f"{body}\n\n{cheer}\n{FALLBACK_NOTE}"
    body = _ensure_citations(body, docs)
    return f"{body}\n\n{cheer}"


async def _complete(prompt: str, question: str, api_key: str) -> str:
    response = await get_llm(api_key).ainvoke(
        [("system", prompt), ("human", question)]
    )
    return _as_text(response.content)


def _clean_rewrite(raw: str) -> str:
    line = next((part.strip() for part in raw.splitlines() if part.strip()), "")
    if "→" in line:
        line = line.split("→")[-1]
    line = line.strip(" «»\"'`“”")
    line = re.sub(r"^(іздеу\s*(сұрағы|жолы)?\s*[:：]\s*)", "", line, flags=re.IGNORECASE)
    return " ".join(line.split())


async def rewrite_query(question: str, api_key: str) -> str:
    """Expand the question into search keywords. Failure keeps the original."""
    global _last_search_query
    original = " ".join(question.split())
    try:
        rewritten = _clean_rewrite(await _complete(REWRITE_PROMPT, original, api_key))
        if len(rewritten) < 2 or len(rewritten) > 240:
            raise ValueError("unusable rewrite")
        _last_search_query = rewritten
        return rewritten
    except Exception:
        _last_search_query = original
        return original


def _support_indexes(raw: str, count: int) -> list[int]:
    if re.fullmatch(r"\s*0\s*", raw):
        return []
    indexes: list[int] = []
    for item in re.findall(r"\b(\d{1,2})\b", raw):
        value = int(item)
        if value == 0 or not 1 <= value <= count or value in indexes:
            continue
        indexes.append(value)
        if len(indexes) == 4:
            break
    return indexes


def _choice_index(raw: str, count: int) -> int | None:
    stripped = raw.strip()
    if re.fullmatch(r"\d{1,2}", stripped):
        value = int(stripped)
        if 1 <= value <= count:
            return value
    labeled = re.search(r"(?:жауап|үзінді|нөмір)\s*[:：]?\s*(\d{1,2})", raw, re.IGNORECASE)
    if labeled:
        value = int(labeled.group(1))
        if 1 <= value <= count:
            return value
    numbers = [int(item) for item in re.findall(r"\b(\d{1,2})\b", raw)]
    in_range = [value for value in numbers if 1 <= value <= count]
    if len(in_range) == 1:
        return in_range[0]
    return None


def _focus_snippet(text: str, terms: list[str]) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    at = -1
    for term in terms:
        at = low.find(term)
        if at >= 0:
            break
    if at < 0:
        return flat[:700]
    return flat[max(0, at - 180) : at + 520]


def _not_found_answer() -> str:
    return f"🌍 📚\n{NOT_FOUND_FACT}"


_YEAR_TOKEN = re.compile(r"\b((?:19|20)\d{2})\b")
_WHEN_ASK = re.compile(
    r"қай\s+жыл|қашан|ашыл|құрыл|ұйымдастыр|негізі\s+қалан",
    re.IGNORECASE,
)
_YEAR_STOP = _FOCUS_STOP | _GENERIC_ASK | {
    "қай",
    "жыл",
    "жылы",
    "жылының",
    "ашылған",
    "ашылды",
    "ашылуы",
    "құрылған",
    "құрылды",
    "ұйымдастырылған",
    "ұйымдастырылды",
    "негізі",
    "қаланған",
    "қаланды",
    "қашан",
}


def _object_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", question.lower()):
        if word in _YEAR_STOP or word in seen:
            continue
        seen.add(word)
        terms.append(word)
    return terms


def _year_near(text: str, terms: list[str]) -> tuple[str, int] | None:
    """Year written next to the asked name. A distant year in the same chunk is ignored."""
    if not terms:
        return None
    low = _dehyphen(text).lower()
    longest = max(terms, key=len)
    if longest not in low:
        return None
    others = [term for term in terms if term != longest]
    if others and not any(term[:5] in low for term in others):
        return None
    anchors = [match.start() for match in re.finditer(re.escape(longest), low)]
    best_year: str | None = None
    best_score = 10**9
    for match in _YEAR_TOKEN.finditer(low):
        dist = min(abs(match.start() - anchor) for anchor in anchors)
        if dist > 420:
            continue
        window = low[max(0, match.start() - 140) : match.end() + 240]
        founded = any(mark in window for mark in ("құрыл", "ашыл", "ұйымдаст", "негіз"))
        score = dist - (250 if founded else 0)
        if score < best_score:
            best_score = score
            best_year = match.group(1)
    if best_year is None:
        return None
    return best_year, best_score


def _year_fact(question: str, docs: list) -> tuple[str, list]:
    """Year next to the asked object. Empty when none of the chunks state one."""
    if not _WHEN_ASK.search(question):
        return "", []
    terms = _object_terms(question)
    found: list[tuple[str, int, Document]] = []
    for doc in docs:
        hit = _year_near(doc.page_content, terms)
        if hit:
            found.append((hit[0], hit[1], doc))
    if not found:
        return "", []
    best_score: dict[str, int] = {}
    docs_for: dict[str, list] = {}
    for year, score, doc in found:
        docs_for.setdefault(year, []).append(doc)
        if year not in best_score or score < best_score[year]:
            best_score[year] = score
    year = min(best_score, key=best_score.get)
    return year, docs_for[year]


def _named_phrase(question: str) -> str:
    kept: list[str] = []
    for word in question.split():
        bare = word.strip("?.,!;:«»\"'").lower()
        if bare in _YEAR_STOP:
            continue
        kept.append(word.strip("?"))
    return " ".join(kept).strip() or " ".join(question.split())


_TOPIC_STOP = _YEAR_STOP | {
    "ең",
    "еңбек",
    "түр",
    "түрі",
    "түрлері",
    "түрлерін",
    "қандай",
    "деген",
    "бар",
    "болып",
    "не",
}


def _content_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", question.lower()):
        if word in _TOPIC_STOP or word in seen:
            continue
        stem = word if len(word) <= 5 else word[:5]
        if stem in seen:
            continue
        seen.add(stem)
        terms.append(stem)
    return terms


def _about_topic(question: str, docs: list) -> bool:
    """True when a retrieved chunk shares the topic. An exact keyword is not required."""
    terms = _content_terms(question)
    extras: list[str] = []
    for pattern, synonyms in _SYNONYM_GROUPS:
        if pattern.search(question):
            extras.extend(synonym.lower() for synonym in synonyms)
    if not terms and not extras:
        return bool(docs)
    for doc in docs:
        low = _dehyphen(doc.page_content).lower()
        if any(term in low for term in terms):
            return True
        if any(extra in low for extra in extras):
            return True
        if _YEAR_TOKEN.search(low) and any(term[:4] in low for term in terms):
            return True
    return False


def _best_sentence(question: str, docs: list) -> str:
    terms = _content_terms(question)
    for pattern, synonyms in _SYNONYM_GROUPS:
        if pattern.search(question):
            terms.extend(synonym.lower()[:5] for synonym in synonyms)
    best = ""
    best_hits = 0
    for doc in docs:
        flat = _dehyphen(doc.page_content)
        for sentence in re.split(r"(?<=[.!?])\s+", flat):
            low = sentence.lower()
            hits = sum(1 for term in terms if term and term in low)
            if hits > best_hits and len(sentence) >= 40:
                best_hits = hits
                best = " ".join(sentence.split())
    return best.strip()


def _cite_docs(docs: list, answer: str, preferred: list) -> list:
    ordered: list = []
    seen: set[tuple] = set()

    def add(doc: Document) -> None:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            return
        seen.add(key)
        ordered.append(doc)

    for doc in preferred:
        add(doc)
        if len(ordered) >= 4:
            return ordered
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{5,}", answer)
        if word.lower() not in _TOPIC_STOP
    ]
    ranked: list[tuple[int, Document]] = []
    for doc in docs:
        low = _dehyphen(doc.page_content).lower()
        ranked.append((sum(1 for word in words if word[:5] in low), doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for hits, doc in ranked:
        if hits <= 0:
            break
        add(doc)
        if len(ordered) >= 4:
            break
    if not ordered:
        for doc in docs[:4]:
            add(doc)
    return ordered[:4]


def _merge_docs(first: list, second: list) -> list:
    merged = []
    seen: set[tuple] = set()
    for doc in list(first) + list(second):
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def _supporting_docs(
    question: str, found: list[tuple[Document, float]]
) -> list[Document]:
    """Passages whose text actually answers the question, at most four."""
    kind = _list_kind(question)
    if not kind:
        return []
    topics = _topic_terms(question)
    docs = []
    for doc, _score in found:
        names = _strict_names(question, [doc], kind)
        if not names:
            continue
        if kind == "country":
            text = _dehyphen(doc.page_content).lower()
            kept = []
            for name in names:
                at = text.find(name.lower())
                window = text[max(0, at - 220) : at + len(name) + 220] if at >= 0 else ""
                if any(topic in window for topic in topics) or "түбек" in window:
                    kept.append(name)
                elif name.lower() == "италия":
                    kept.append(name)
            if not kept:
                continue
        docs.append(doc)
        if len(docs) == 4:
            break
    return docs


def _focused_docs(
    search_query: str, found: list[tuple[Document, float]]
) -> list[Document]:
    terms = _focus_terms(search_query)
    if not terms:
        return []
    rare = max(terms, key=len)
    rare_docs = [
        doc
        for doc, _score in found
        if rare in _dehyphen(doc.page_content).lower()
    ]
    return rare_docs[:4]


def _plain_from_names(question: str, names: list[str]) -> str:
    joined = ", ".join(names)
    kind = _list_kind(question)
    if kind == "country":
        return f"Оқулықта осы сұраққа сай келетін ел — {joined}."
    if kind == "river":
        return f"Оқулық бойынша бұл су нысаны Каспийге қатысты аталады: {joined}."
    if kind == "deposit":
        return f"Оқулықта аталған кен орны — {joined}."
    return f"Оқулық бойынша жауап: {joined}."


async def pick_supporting(
    question: str,
    found: list[tuple[Document, float]],
    api_key: str,
    search_query: str,
) -> list[Document]:
    grounded = _supporting_docs(question, found)
    if grounded:
        return grounded
    if _list_kind(question):
        return []
    focused = _focused_docs(search_query, found)
    if focused:
        return focused
    terms = _focus_terms(search_query)
    snippets = [
        f"{number}. {_focus_snippet(doc.page_content, terms)}"
        for number, (doc, _score) in enumerate(found, start=1)
    ]
    try:
        raw = await _complete(
            RERANK_PROMPT,
            f"Сұрақ: {question}\n\n" + "\n\n".join(snippets),
            api_key,
        )
        indexes = _support_indexes(raw, len(found))
    except Exception:
        indexes = []
    if not indexes:
        return [max(found, key=lambda item: item[1])[0]]
    return [found[index - 1][0] for index in indexes]


def _styled_explanation(answer: str, docs: list) -> str:
    kept = [
        line
        for line in answer.splitlines()
        if line.strip() != FALLBACK_NOTE and line.strip() not in ENCOURAGEMENTS
    ]
    body = re.sub(r"\s*\[Дереккөз:[^\]]*\]\.?", "", "\n".join(kept)).strip()
    if NOT_FOUND_FACT in body:
        rest = body.replace(NOT_FOUND_FACT, "").strip(" \n.,;")
        if len(rest) < 20:
            return _not_found_answer()
        body = rest
    first = body.splitlines()[0]
    if not any(emoji in first for emoji in TOPIC_EMOJIS):
        body = f"🌍 📚\n{body}"
    body = _ensure_citations(body, docs[:4])
    return f"{body}\n\n{_next_encouragement()}"


async def answer_question(question: str, api_key: str) -> str:
    search_query = await rewrite_query(question, api_key)
    found = await asyncio.to_thread(
        candidate_pool, search_query, api_key, RAG_TOP_K, question
    )
    if found is None:
        return MISSING_DB
    if not found:
        return _not_found_answer()
    pool_docs = [doc for doc, _score in found]
    fact_year, year_docs = _year_fact(question, pool_docs)
    # The whole semantic pool, so a date or a mineral fact on a nearby page stays visible.
    context_docs = _merge_docs(year_docs, pool_docs)[:RAG_TOP_K]
    names: list[str] = []
    kind = _list_kind(question)
    if kind:
        names = _strict_names(question, context_docs, kind)
    hint = ""
    if names:
        hint = (
            "Тек мына атауларды қолдан, басқа ел мен нысанды қоспа: "
            + ", ".join(names)
            + ". Үзіндіде жоқ географиялық тұжырымды қоспа.\n"
        )
    if fact_year:
        hint += (
            f"Үзіндіде {fact_year} жылы деп жазылған. Жауапта осы жыл тұрсын. "
            "Нақты мәлімет табылмады деп жазба.\n"
        )
    text = await _complete(
        TEXTBOOK_PROMPT,
        f"{hint}Оқулық үзінділері:\n{_context_block(context_docs)}\n\nСұрақ: {question}",
        api_key,
    )
    on_topic = _about_topic(question, context_docs)
    if fact_year and year_docs:
        if NOT_FOUND_FACT in text or fact_year not in text:
            text = f"{_named_phrase(question)} {fact_year} жылы құрылған."
    elif NOT_FOUND_FACT in text:
        if not on_topic:
            return _not_found_answer()
        text = await _complete(
            SYNTHESIZE_PROMPT,
            f"Оқулық үзінділері:\n{_context_block(context_docs)}\n\nСұрақ: {question}",
            api_key,
        )
        if NOT_FOUND_FACT in text:
            text = _best_sentence(question, context_docs)
        if not text:
            return _not_found_answer()
    elif kind == "country" and names:
        allowed = {name.lower() for name in names}
        for country in sorted(_COUNTRIES, key=len, reverse=True):
            if country.lower() in allowed:
                continue
            text = re.sub(
                rf"(?<![\w]){re.escape(country)}{_COUNTRY_SUFFIX}(?![\w])",
                "",
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+,", ",", text)
        if not re.search(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{4,}", text) and on_topic:
            text = _plain_from_names(question, names)
    cite_docs = _cite_docs(context_docs, text, year_docs if fact_year else [])
    return _styled_explanation(text, cite_docs)


THINKING = "💭 ойланып жатырмын бейби..."
WELCOME_GIF = ROOT / "assets" / "geoai_welcome.gif"
WELCOME_FILE_ID = ROOT / "data" / "welcome_file_id.txt"
QUESTIONS_PATH = ROOT / "questions.json"
_database_url = os.getenv("DATABASE_URL", "").strip()
QUIZ_DB = Path(_database_url) if _database_url else ROOT / "data" / "quiz.sqlite"
ALMATY = ZoneInfo("Asia/Almaty")
FREE_POOL = 5
ADMIN_ID = "8091822446"
BTN_ASK = "💬 Сұрақ қою"
BTN_QUIZ = "🎯 Сұраққа жауап беру"
BTN_BEGIN = "Бастау"
BTN_MENU = "Мәзір"
BTN_HOME = "🏠 Басты мәзір"
QUIZ_RETRY = "❌ Қате! Екінші мүмкіндігің бар, тағы бір рет ойланып жазып көр! 💡"
QUIZ_INTRO = (
    "Географиядан 45+ алмай, грантқа түсу мүмкін емес 🎯.\n"
    "Курстарға 30 000 ₸ шашпай-ақ, осы жердегі 3000 сұрақты тас-талқан етсең — "
    "ҰБТ сен үшін ойын болады. Дайынсың ба? Бастаймыз! 💥"
)
ASK_INTRO = (
    "Google таппаған, тестте келген сұрақты маған таста 🔍.\n"
    "19 оқулықтың қай бұрышында жатса да, 1 секундта қай бетте екенін фактымен "
    "бетіне басып берем. Қандай сұрақ бар? ⌨️"
)
FREE_LIMIT_LINE = "Сенде тегін 5 сұрақ лимиті бар."
PRO_COMPARE = (
    "🔒 Толық база жабық VIP-каналда!\n"
    "Жеке репетитор — 25 000 ₸/ай ❌\n"
    "Курстар — 35 000 ₸/ай ❌\n"
    "GeoAI Премиум (19 оқулық + 3000 сұрақтық база) — бары-жоғы 4 990 ₸ ✅"
)
LIMIT_CLOSED = (
    "🔒 Сенің тегін лимитің аяқталды! 3000+ сұрақ пен 19 оқулықтың базасын "
    "толық ашу үшін VIP каналға қосыл."
)
FREE_EMPTY = f"{LIMIT_CLOSED}\n\n{PRO_COMPARE}"
SECTION_CLOSED = (
    "🔒 Тегін лимитің (5/5) аяқталды!\n"
    "\n"
    "ҰБТ & Олимпиадада 19 оқулықтың базасын толық ашып, 1500+ сұрақ пен AI түсіндірмелерге толық доступ алу үшін VIP-ке қосыл 🚀"
)
RECEIPT_PROMPT = "Төлем чегінің скриншотын осында жіберіңіз:"
CARD_DIGITS = "4400430045714687"
PHONE_NUMBER = "+77058951332"
PRO_BENEFITS = (
    "📚 боттағы тегін материалдардың барлығы\n"
    "🎯 1500+ нағыз ұбтда кездесетін сұрақтар викторинасы\n"
    "📝 7-11 сынып толық Quizzes\n"
    "🗺️ Авторлық интерактивті карта жаттау сайты\n"
    "🧮 Авторлық математикалық сауаттылық дайындық сайты\n"
    "♾️ GeoAI ботына шексіз лимит\n"
    "👑 Ашық VIP community"
)
_START_GREETING = (
    "Сәлем! 🚀 Мен — <b>GeoAI</b> ассистентімін.\n"
    "География және ҰБТ сұрақтары бойынша кез келген сұрағыңды қой, "
    "оқулықтар негізінде дәл жауап беремін! 📚"
)
JUDGE_PROMPT = (
    "Сен қазақ тіліндегі география жауабын тексересің.\n"
    "Тек бір JSON қайтар, басқа сөз жоқ:\n"
    '{"verdict":"correct","found":0,"left":0,"explanation":"..."}\n'
    "verdict тек correct, partial немесе incorrect.\n"
    "Ресми жауапты жеке қажетті элементтерге бөл: атау, сан, ел, жыл.\n"
    "Мағынаны бағала. Емле, тыныс белгі, регистр және жақын синоним дұрыс.\n"
    "correct: барлық қажетті элемент бар. Алдыңғы әрекет берілсе, екі әрекетті қосып сана. "
    "Жетпеген элемент осы әрекетте жазылса, correct.\n"
    "partial: қажетті элемент кемінде екеу, оқушы оның біразын тапты, біреуі әлі жоқ. "
    "found — табылған саны, left — қалған саны.\n"
    "incorrect: бірде-бір қажетті элемент жоқ немесе факт қате.\n"
    "Бір ғана қажетті элемент болса, partial қолданба.\n"
    "explanation: сол географиялық түйінді 1 немесе 2 қазақ сөйлемі. "
    "Тек сұрақ пен ресми жауапқа сүйен. Дереккөз, файл, бет және жақшалы дереккөз жазба."
)

_quiz_lock = threading.Lock()
_section_flight: set[tuple[str, str]] = set()
_section_flight_guard = threading.Lock()
_quiz_bank: list[dict] | None = None


def _read_welcome_file_id() -> str:
    try:
        return WELCOME_FILE_ID.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_welcome_file_id(file_id: str) -> None:
    cleaned = file_id.strip()
    if not cleaned:
        return
    WELCOME_FILE_ID.parent.mkdir(parents=True, exist_ok=True)
    WELCOME_FILE_ID.write_text(cleaned + "\n", encoding="utf-8")


def _sent_file_id(sent: Message) -> str:
    if sent.animation is not None and sent.animation.file_id:
        return sent.animation.file_id
    if sent.video is not None and sent.video.file_id:
        return sent.video.file_id
    if sent.document is not None and sent.document.file_id:
        return sent.document.file_id
    return ""


async def _deliver_answer(pending: Message, text: str) -> None:
    """Put the answer in the thinking message. Overflow goes out as a follow-up."""
    body = text or "Жауап бос келді."
    chunks = [body[start : start + TELEGRAM_LIMIT] for start in range(0, len(body), TELEGRAM_LIMIT)]
    head, rest = chunks[0], chunks[1:]
    try:
        await pending.edit_text(head, reply_markup=None if rest else _home_keyboard())
    except Exception:
        await pending.answer(head, reply_markup=None if rest else _home_keyboard())
    for index, chunk in enumerate(rest):
        last = index == len(rest) - 1
        await pending.answer(chunk, reply_markup=_home_keyboard() if last else None)


def _menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_QUIZ, callback_data="mode:quiz")],
            [InlineKeyboardButton(text=BTN_ASK, callback_data="mode:ask")],
            [InlineKeyboardButton(text="🤖 Негізгі ботқа өту", url="https://t.me/geomachinebot")],
        ]
    )


def _home_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(text=BTN_HOME, callback_data="menu:home")


def _home_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_home_button()]])


def _next_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➡️ Келесі сұрақ", callback_data="next_q")],
            [_home_button()],
        ]
    )


def _with_home(markup: InlineKeyboardMarkup) -> InlineKeyboardMarkup:
    rows = [list(row) for row in markup.inline_keyboard]
    rows.append([_home_button()])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _cap_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_MENU, callback_data="menu:open")],
        ]
    )


def _begin_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_BEGIN, callback_data="quiz:begin")],
        ]
    )


def _upgrade_keyboard() -> InlineKeyboardMarkup:
    return _with_home(
        InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🌟 PRO деңгейге өту", callback_data="pay:open")],
            ]
        )
    )


def _pay_keyboard() -> InlineKeyboardMarkup:
    return _with_home(
        InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔴 Kaspi-де төлеу", url="https://kaspi.kz/")],
                [InlineKeyboardButton(text="✅ Төледім", callback_data="pay:done")],
                [InlineKeyboardButton(text="🟣 PRO деңгейде не бар?", callback_data="pay:benefits")],
                [InlineKeyboardButton(text="⬅️ Артқа", callback_data="menu:open")],
            ]
        )
    )


def _benefits_keyboard() -> InlineKeyboardMarkup:
    return _with_home(
        InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Артқа", callback_data="pay:open")],
            ]
        )
    )


def _private_channel_url() -> str:
    return os.getenv("PRIVATE_CHANNEL_URL", "").strip()


def _pro_confirm_text() -> str:
    return "PRO деңгейіңіз іске қосылды. Сұрақ қою мен викторина енді шексіз."


def _pay_html() -> str:
    """Card and phone sit in code tags so Telegram offers tap-to-copy."""
    return "\n".join(
        [
            html.escape(PRO_COMPARE),
            "",
            html.escape("⭐ PRO деңгей — 4990 тг"),
            html.escape("Реквизиттер (Kaspi):"),
            html.escape("💳 Карта: ") + f"<code>{CARD_DIGITS}</code>",
            html.escape("👤 Алушы: Ақжол Әйлеш"),
            html.escape("📱 Телефон: ") + f"<code>{PHONE_NUMBER}</code>",
            html.escape("Kaspi.kz → Переводы → Клиенту Kaspi → По номеру телефона"),
            html.escape("1) «Kaspi-де төлеу» батырмасы Kaspi сайтына апарады."),
            html.escape(
                "2) Төлемді нөмір арқылы жасап болған соң, «Төледім» батырмасын басып чекті жіберіңіз."
            ),
        ]
    )


async def _edit_frames(sent: Message, frames: tuple[str, ...]) -> None:
    """Two or three short edits, then the caller shows the final text."""
    for frame in frames:
        await asyncio.sleep(0.5)
        try:
            await sent.edit_text(frame)
        except Exception:
            return


def _load_bank() -> list[dict]:
    global _quiz_bank
    if _quiz_bank is None:
        raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        _quiz_bank = [item for item in raw if item.get("question") and item.get("answer")]
    return _quiz_bank


def _bank_by_id() -> dict[int, dict]:
    return {int(item["id"]): item for item in _load_bank()}


def _migrate_ordered_progress(conn: sqlite3.Connection) -> None:
    """Turn an old file-order index into seen ids plus a started count."""
    bank = _load_bank()
    rows = conn.execute("SELECT user_id, q_index, shown FROM progress").fetchall()
    for user_id, q_index, shown in rows:
        q_index = int(q_index or 0)
        shown = int(shown or 0)
        if q_index <= 0 and not shown:
            continue
        last = q_index if shown else q_index - 1
        if last < 0 or not bank:
            continue
        last = min(last, len(bank) - 1)
        delivered = bank[: last + 1]
        conn.executemany(
            "INSERT OR IGNORE INTO seen (user_id, question_id) VALUES (?, ?)",
            [(user_id, int(item["id"])) for item in delivered],
        )
        current_id = int(delivered[-1]["id"]) if shown else None
        conn.execute(
            "UPDATE progress SET q_index = ?, current_id = ? WHERE user_id = ?",
            (len(delivered), current_id, user_id),
        )


def _quiz_connect() -> sqlite3.Connection:
    QUIZ_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUIZ_DB, timeout=30, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            user_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'ask',
            q_index INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            score INTEGER NOT NULL DEFAULT 0,
            shown INTEGER NOT NULL DEFAULT 0,
            current_id INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily (
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            sent INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, day)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen (
            user_id TEXT NOT NULL,
            question_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, question_id)
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(progress)")}
    if "shown" not in columns:
        conn.execute("ALTER TABLE progress ADD COLUMN shown INTEGER NOT NULL DEFAULT 0")
    if "current_id" not in columns:
        conn.execute("ALTER TABLE progress ADD COLUMN current_id INTEGER")
        _migrate_ordered_progress(conn)
    if "first_try" not in columns:
        conn.execute("ALTER TABLE progress ADD COLUMN first_try TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS access (
            user_id TEXT PRIMARY KEY,
            used INTEGER NOT NULL DEFAULT 0,
            pro INTEGER NOT NULL DEFAULT 0,
            await_receipt INTEGER NOT NULL DEFAULT 0,
            quiz_used INTEGER NOT NULL DEFAULT 0,
            is_vip INTEGER NOT NULL DEFAULT 0,
            vip_expires_at TEXT
        )
        """
    )
    access_columns = {row[1] for row in conn.execute("PRAGMA table_info(access)")}
    if "quiz_used" not in access_columns:
        conn.execute(
            "ALTER TABLE access ADD COLUMN quiz_used INTEGER NOT NULL DEFAULT 0"
        )
    if "is_vip" not in access_columns:
        conn.execute("ALTER TABLE access ADD COLUMN is_vip INTEGER NOT NULL DEFAULT 0")
    if "vip_expires_at" not in access_columns:
        conn.execute("ALTER TABLE access ADD COLUMN vip_expires_at TEXT")
    conn.execute("UPDATE access SET is_vip = 1 WHERE pro = 1 AND is_vip = 0")
    conn.commit()
    return conn


def _quiz_row(conn: sqlite3.Connection, user_id: str) -> dict:
    row = conn.execute(
        """
        SELECT mode, q_index, attempts, score, shown, current_id, first_try
        FROM progress WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO progress
                (user_id, mode, q_index, attempts, score, shown, current_id, first_try)
            VALUES (?, 'ask', 0, 0, 0, 0, NULL, '')
            """,
            (user_id,),
        )
        conn.commit()
        return {
            "mode": "ask",
            "q_index": 0,
            "attempts": 0,
            "score": 0,
            "shown": 0,
            "current_id": None,
            "first_try": "",
        }
    return {
        "mode": row[0],
        "q_index": int(row[1]),
        "attempts": int(row[2]),
        "score": int(row[3]),
        "shown": int(row[4] or 0),
        "current_id": None if row[5] is None else int(row[5]),
        "first_try": row[6] or "",
    }


def _leave_special_modes(conn: sqlite3.Connection, user_id: str) -> None:
    """Stop treating the next text as a quiz answer or the next photo as a receipt."""
    state = _quiz_row(conn, user_id)
    state["mode"] = "ask"
    _save_progress(conn, user_id, state)
    _set_await_receipt(conn, user_id, False)


def _save_progress(conn: sqlite3.Connection, user_id: str, state: dict) -> None:
    conn.execute(
        """
        UPDATE progress
        SET mode = ?, q_index = ?, attempts = ?, score = ?, shown = ?, current_id = ?, first_try = ?
        WHERE user_id = ?
        """,
        (
            state["mode"],
            int(state["q_index"]),
            int(state["attempts"]),
            int(state["score"]),
            int(state["shown"]),
            state.get("current_id"),
            state.get("first_try") or "",
            user_id,
        ),
    )
    conn.commit()


def _seen_ids(conn: sqlite3.Connection, user_id: str) -> set[int]:
    rows = conn.execute(
        "SELECT question_id FROM seen WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    return {int(row[0]) for row in rows}


def _remember_seen(conn: sqlite3.Connection, user_id: str, question_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen (user_id, question_id) VALUES (?, ?)",
        (user_id, int(question_id)),
    )


def _pick_unseen(conn: sqlite3.Connection, user_id: str, bank: list[dict]) -> dict | None:
    seen = _seen_ids(conn, user_id)
    unseen = [item for item in bank if int(item["id"]) not in seen]
    if not unseen:
        return None
    return random.choice(unseen)


def _almaty_day() -> str:
    return datetime.now(ALMATY).date().isoformat()


def _ensure_access(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO access (user_id, used, pro, await_receipt) VALUES (?, 0, 0, 0)",
        (user_id,),
    )


def _expiry_passed(value: object) -> bool:
    if value is None or str(value).strip() == "":
        return False
    text = str(value).strip().replace("Z", "+00:00")
    try:
        expires = datetime.fromisoformat(text)
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires < datetime.now(timezone.utc)


def _vip_active(conn: sqlite3.Connection, user_id: str) -> bool:
    """Active VIP. NULL vip_expires_at is lifetime. A past expiry is cleared."""
    _ensure_access(conn, user_id)
    row = conn.execute(
        "SELECT is_vip, vip_expires_at FROM access WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row is None or not int(row[0] or 0):
        return False
    if _expiry_passed(row[1]):
        conn.execute(
            "UPDATE access SET is_vip = 0, pro = 0 WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
        return False
    return True


def _is_pro(conn: sqlite3.Connection, user_id: str) -> bool:
    return _vip_active(conn, user_id)


def _unlimited_user(conn: sqlite3.Connection, user_id: str) -> bool:
    return _is_owner_id(user_id) or _vip_active(conn, user_id)


def _show_free_limit_line(user_id: object) -> bool:
    """Free users who can still start see the 5-request line. Owner and PRO do not."""
    if user_id is None or _is_owner_id(user_id):
        return False
    key = str(user_id).strip()
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            return not _is_pro(conn, key)
        finally:
            conn.close()


def _with_free_limit_line(intro: str, user_id: object) -> str:
    if _show_free_limit_line(user_id):
        return f"{intro}\n\n{FREE_LIMIT_LINE}"
    return intro


def _ask_intro_for(user_id: object) -> str:
    return _with_free_limit_line(ASK_INTRO, user_id)


def _quiz_intro_for(user_id: object) -> str:
    return _with_free_limit_line(QUIZ_INTRO, user_id)


def _section_column(section: str) -> str:
    """Ask keeps the old shared `used` column. Quiz has its own counter."""
    if section == "quiz":
        return "quiz_used"
    return "used"


def _section_used(conn: sqlite3.Connection, user_id: str, section: str) -> int:
    """Persisted count. A missing row is zero and is not written back as a reset."""
    column = _section_column(section)
    row = conn.execute(
        f"SELECT {column} FROM access WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def _section_blocked(conn: sqlite3.Connection, user_id: str, section: str) -> bool:
    """Owner and PRO have no cap. Each section has its own FREE_POOL."""
    if _unlimited_user(conn, user_id):
        return False
    return _section_used(conn, user_id, section) >= FREE_POOL


def _stored_section_blocked(user_id: str, section: str) -> bool:
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            return _section_blocked(conn, user_id, section)
        finally:
            conn.close()


def _release_section(user_id: str, section: str) -> None:
    with _section_flight_guard:
        _section_flight.discard((user_id, section))


def _claim_section(user_id: str, section: str) -> str:
    """Atomic cap. 'ok' may proceed, 'blocked' is the banner, 'busy' is a duplicate tap."""
    key = (user_id, section)
    with _section_flight_guard:
        if key in _section_flight:
            return "busy"
        _section_flight.add(key)
    conn = _quiz_connect()
    try:
        if _is_owner_id(user_id) or _vip_active(conn, user_id):
            return "ok"
        _ensure_access(conn, user_id)
        column = _section_column(section)
        before = conn.total_changes
        conn.execute(
            f"UPDATE access SET {column} = {column} + 1 "
            f"WHERE user_id = ? AND {column} < ?",
            (user_id, FREE_POOL),
        )
        conn.commit()
        if conn.total_changes - before == 1:
            return "ok"
        _release_section(user_id, section)
        return "blocked"
    except Exception:
        _release_section(user_id, section)
        raise
    finally:
        conn.close()


def _awaiting_receipt(conn: sqlite3.Connection, user_id: str) -> bool:
    _ensure_access(conn, user_id)
    row = conn.execute(
        "SELECT await_receipt FROM access WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return bool(row and int(row[0]))


def _set_await_receipt(conn: sqlite3.Connection, user_id: str, waiting: bool) -> None:
    _ensure_access(conn, user_id)
    conn.execute(
        "UPDATE access SET await_receipt = ? WHERE user_id = ?",
        (1 if waiting else 0, user_id),
    )
    conn.commit()


def _grant_pro(conn: sqlite3.Connection, user_id: str) -> None:
    _ensure_access(conn, user_id)
    conn.execute(
        """
        UPDATE access
        SET pro = 1, is_vip = 1, vip_expires_at = NULL, await_receipt = 0
        WHERE user_id = ?
        """,
        (user_id,),
    )
    conn.commit()


def _vip_offer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💳 VIP Доступ алу (4 990 ₸)", callback_data="pay:open")],
        ]
    )


async def _answer_free_empty(message: Message) -> None:
    await message.answer(SECTION_CLOSED, reply_markup=_vip_offer_keyboard())


async def _send_pay_screen(message: Message) -> None:
    await message.answer(
        _pay_html(),
        parse_mode=ParseMode.HTML,
        reply_markup=_pay_keyboard(),
    )


def _daily_sent(conn: sqlite3.Connection, user_id: str) -> int:
    row = conn.execute(
        "SELECT sent FROM daily WHERE user_id = ? AND day = ?",
        (user_id, _almaty_day()),
    ).fetchone()
    return int(row[0]) if row else 0


def _mark_sent(conn: sqlite3.Connection, user_id: str) -> int:
    day = _almaty_day()
    conn.execute(
        """
        INSERT INTO daily (user_id, day, sent) VALUES (?, ?, 1)
        ON CONFLICT(user_id, day) DO UPDATE SET sent = sent + 1
        """,
        (user_id, day),
    )
    conn.commit()
    return _daily_sent(conn, user_id)


def _format_quiz_item(item: dict, started: int, total: int) -> str:
    """started is how many questions this user has opened, not the file id."""
    return f"🎯 {started}-сұрақ\n\n{item['question']}"


_SOURCE_BRACKET = re.compile(
    r"\[[^\]\n]*(?:Дереккөз|дереккөз|бет\s*\d|source)[^\]\n]*\]",
    re.IGNORECASE,
)


def _md_escape(text: str) -> str:
    return re.sub(r"([_*`\[])", r"\\\1", text or "")


def _reveal_answer(answer: str, explanation: str) -> str:
    return (
        f"❌ **Дұрыс жауап: {_md_escape(_quiz_plain(answer))}**\n\n"
        f"{_md_escape(explanation)}"
    )


def _quiz_plain(text: str) -> str:
    """Quiz replies never carry a source bracket."""
    cleaned = _SOURCE_BRACKET.sub("", text or "")
    cleaned = re.sub(r"\[\s*Дереккөз:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _quiz_note(text: str) -> str:
    blob = _quiz_plain(text)
    blob = re.sub(r"\s+", " ", blob).strip()
    parts = [part for part in re.split(r"(?<=[.!?])\s+", blob) if part]
    return " ".join(parts[:2]).strip()


def _fallback_note(item: dict) -> str:
    return f"{_quiz_plain(str(item.get('question') or ''))} — {_quiz_plain(str(item.get('answer') or ''))}."


def _partial_line(found: int, left: int) -> str:
    return (
        f"💡 Жарайсың! {found} дұрыс жауапты таптың, "
        f"тағы {left} дұрыс жауап қалды. Қалғанын да жазып көр!"
    )


def _as_count(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _parse_grade(raw: str) -> dict | None:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = None
    if match:
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            data = loaded
    if data is None:
        if re.search(r"\bincorrect\b", text, re.IGNORECASE):
            return {"verdict": "incorrect", "found": 0, "left": 0, "explanation": ""}
        if re.search(r"\bcorrect\b", text, re.IGNORECASE):
            return {"verdict": "correct", "found": 0, "left": 0, "explanation": ""}
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    found = _as_count(data.get("found"))
    left = _as_count(data.get("left"))
    explanation = _quiz_note(str(data.get("explanation") or ""))
    if verdict in {"correct", "дұрыс", "true", "yes"}:
        return {"verdict": "correct", "found": found, "left": 0, "explanation": explanation}
    if verdict in {"partial", "жартылай"}:
        if found >= 1 and left >= 1:
            return {
                "verdict": "partial",
                "found": found,
                "left": left,
                "explanation": explanation,
            }
        if found >= 1:
            return {"verdict": "correct", "found": found, "left": 0, "explanation": explanation}
        return {"verdict": "incorrect", "found": 0, "left": left, "explanation": explanation}
    if verdict in {"incorrect", "қате", "false", "no"}:
        return {"verdict": "incorrect", "found": 0, "left": left, "explanation": explanation}
    return None


async def _judge_answer(
    item: dict, user_text: str, api_key: str, previous: str = ""
) -> dict | None:
    """Meaning check. Gemini decides; the strings are not compared."""
    lines = [
        f"Сұрақ: {item['question']}",
        f"Ресми жауап: {item['answer']}",
    ]
    earlier = previous.strip()
    if earlier:
        lines.append(f"Алдыңғы әрекет: {earlier}")
        lines.append(f"Осы әрекет: {user_text}")
    else:
        lines.append(f"Оқушы жауабы: {user_text}")
    raw = await _complete(JUDGE_PROMPT, "\n".join(lines), api_key)
    return _parse_grade(raw)


async def _send_welcome_animation(
    message: Message, media: str | FSInputFile
) -> Message:
    return await message.answer_animation(
        media,
        caption=_START_GREETING,
        parse_mode=ParseMode.HTML,
        reply_markup=_menu_keyboard(),
    )


def _open_quiz_item(
    conn: sqlite3.Connection, user_id: str, state: dict, spend: bool = True
) -> dict | None:
    """Choose a random unseen question and count it as started."""
    bank = _load_bank()
    item = _pick_unseen(conn, user_id, bank)
    if item is None:
        return None
    question_id = int(item["id"])
    _remember_seen(conn, user_id, question_id)
    state["current_id"] = question_id
    state["q_index"] = int(state["q_index"]) + 1
    state["shown"] = 1
    state["attempts"] = 0
    state["first_try"] = ""
    state["mode"] = "quiz"
    _save_progress(conn, user_id, state)
    return item


def _close_quiz_round(user_id: str, state: dict) -> None:
    """End this question without loading another one."""
    state["attempts"] = 0
    state["shown"] = 0
    state["current_id"] = None
    state["first_try"] = ""
    state["mode"] = "quiz"
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            _save_progress(conn, user_id, state)
        finally:
            conn.close()


async def _send_quiz_question(message: Message, user_id: str) -> None:
    bank = _load_bank()
    by_id = _bank_by_id()
    claimed = False
    try:
        item = None
        state: dict = {}
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                state = _quiz_row(conn, user_id)
                if state["shown"] and state["current_id"] in by_id:
                    item = by_id[int(state["current_id"])]
                    text = _format_quiz_item(item, int(state["q_index"]), len(bank))
                    await message.answer(text, reply_markup=_home_keyboard())
                    return
                if int(state["q_index"]) >= len(bank) or _pick_unseen(conn, user_id, bank) is None:
                    await message.answer(
                        f"Барлық сұрақ аяқталды. Ұпайың: {state['score']}.",
                        reply_markup=_home_keyboard(),
                    )
                    return
                status = _claim_section(user_id, "quiz")
                if status == "busy":
                    return
                if status != "ok":
                    await _answer_free_empty(message)
                    return
                claimed = True
                item = _open_quiz_item(conn, user_id, state, spend=False)
            finally:
                conn.close()
        if item is None:
            await message.answer(
                f"Барлық сұрақ аяқталды. Ұпайың: {state['score']}.",
                reply_markup=_home_keyboard(),
            )
            return
        await message.answer(
            _format_quiz_item(item, int(state["q_index"]), len(bank)),
            reply_markup=_home_keyboard(),
        )
    finally:
        if claimed:
            _release_section(user_id, "quiz")


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    cached = _read_welcome_file_id()
    if cached:
        try:
            await _send_welcome_animation(message, cached)
            return
        except Exception:
            pass
    if WELCOME_GIF.is_file():
        try:
            sent = await _send_welcome_animation(message, FSInputFile(WELCOME_GIF))
            _write_welcome_file_id(_sent_file_id(sent))
            return
        except Exception:
            pass
    await message.answer(
        _START_GREETING,
        parse_mode=ParseMode.HTML,
        reply_markup=_menu_keyboard(),
    )


@router.callback_query(F.data == "menu:home")
async def on_home_menu(query: CallbackQuery) -> None:
    """Leave quiz answers and receipt waiting, then show the two-choice menu."""
    await query.answer()
    sender = query.from_user
    if sender is not None:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                _leave_special_modes(conn, str(sender.id))
            finally:
                conn.close()
    if query.message is None:
        return
    await query.message.answer(
        _START_GREETING,
        parse_mode=ParseMode.HTML,
        reply_markup=_menu_keyboard(),
    )


@router.callback_query(F.data == "menu:open")
async def on_open_menu(query: CallbackQuery) -> None:
    """Same two /start choices, without the Earth animation or a question."""
    await query.answer()
    if query.message is None:
        return
    await query.message.answer(
        _START_GREETING,
        parse_mode=ParseMode.HTML,
        reply_markup=_menu_keyboard(),
    )


@router.callback_query(F.data == "pay:open")
async def on_pay_open(query: CallbackQuery) -> None:
    await query.answer()
    if query.message is not None:
        await _send_pay_screen(query.message)


@router.callback_query(F.data == "pay:benefits")
async def on_pay_benefits(query: CallbackQuery) -> None:
    await query.answer()
    if query.message is None:
        return
    sent = await query.message.answer("✨")
    await _edit_frames(sent, ("✨✨", "✨✨✨"))
    try:
        await sent.edit_text(PRO_BENEFITS, reply_markup=_benefits_keyboard())
    except Exception:
        await query.message.answer(PRO_BENEFITS, reply_markup=_benefits_keyboard())


@router.callback_query(F.data == "pay:done")
async def on_pay_done(query: CallbackQuery) -> None:
    await query.answer()
    sender = query.from_user
    if sender is not None:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                _set_await_receipt(conn, str(sender.id), True)
            finally:
                conn.close()
    if query.message is not None:
        await query.message.answer(RECEIPT_PROMPT, reply_markup=_home_keyboard())


@router.callback_query(F.data.startswith("pro:ok:"))
async def on_pro_approve(query: CallbackQuery) -> None:
    await query.answer()
    admin = query.from_user
    if admin is None or str(admin.id) != ADMIN_ID:
        return
    payer = (query.data or "").split(":")[-1]
    if not payer.isdigit():
        return
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            _grant_pro(conn, payer)
        finally:
            conn.close()
    bot = query.bot
    if bot is not None:
        try:
            sent = await bot.send_message(int(payer), "🎉")
            await _edit_frames(sent, ("🎉✨",))
            await sent.edit_text(_pro_confirm_text(), reply_markup=_home_keyboard())
            invite = _private_channel_url()
            if invite:
                await bot.send_message(int(payer), invite, reply_markup=_home_keyboard())
        except Exception:
            pass
    if query.message is not None:
        await query.message.answer(f"PRO қосылды: {payer}")


@router.callback_query(F.data == "mode:ask")
async def on_ask_mode(query: CallbackQuery) -> None:
    await query.answer()
    sender = query.from_user
    if sender is not None and _stored_section_blocked(str(sender.id), "ask"):
        if query.message is not None:
            await _answer_free_empty(query.message)
        return
    if sender is not None:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                state = _quiz_row(conn, str(sender.id))
                state["mode"] = "ask"
                _save_progress(conn, str(sender.id), state)
            finally:
                conn.close()
    if query.message is not None:
        await query.message.answer(_ask_intro_for(sender.id if sender is not None else None))


@router.callback_query(F.data == "mode:quiz")
async def on_quiz_menu(query: CallbackQuery) -> None:
    """Intro only. The first question waits for Бастау."""
    await query.answer()
    sender = query.from_user
    if sender is not None and _stored_section_blocked(str(sender.id), "quiz"):
        if query.message is not None:
            await _answer_free_empty(query.message)
        return
    if sender is not None:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                state = _quiz_row(conn, str(sender.id))
                state["mode"] = "quiz"
                _save_progress(conn, str(sender.id), state)
            finally:
                conn.close()
    if query.message is not None:
        await query.message.answer(
            _quiz_intro_for(sender.id if sender is not None else None),
            reply_markup=_begin_keyboard(),
        )


@router.callback_query(F.data == "quiz:begin")
async def on_quiz_begin(query: CallbackQuery) -> None:
    await query.answer()
    sender = query.from_user
    if sender is None or query.message is None:
        return
    user_id = str(sender.id)
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            state = _quiz_row(conn, user_id)
            in_progress = bool(state["shown"] and state.get("current_id") is not None)
            if _section_blocked(conn, user_id, "quiz") and not in_progress:
                blocked = True
            else:
                blocked = False
                state["mode"] = "quiz"
                if int(state["q_index"]) == 0 and not state["shown"]:
                    state["attempts"] = 0
                _save_progress(conn, user_id, state)
        finally:
            conn.close()
    if blocked:
        await _answer_free_empty(query.message)
        return
    await _send_quiz_question(query.message, user_id)


@router.callback_query(F.data == "next_q")
async def on_next_question(query: CallbackQuery) -> None:
    """Loads the next quiz question and spends one quiz request, not an ask request."""
    await query.answer()
    sender = query.from_user
    if sender is None or query.message is None:
        return
    await _send_next_question(query.message, str(sender.id))


async def _send_next_question(message: Message, user_id: str) -> None:
    bank = _load_bank()
    by_id = _bank_by_id()
    item = None
    text = ""
    claimed = False
    state: dict = {}
    try:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                state = _quiz_row(conn, user_id)
                if state["shown"] and state["current_id"] in by_id:
                    item = by_id[int(state["current_id"])]
                    text = _format_quiz_item(item, int(state["q_index"]), len(bank))
                elif _pick_unseen(conn, user_id, bank) is None:
                    item = None
                else:
                    status = _claim_section(user_id, "quiz")
                    if status == "busy":
                        return
                    if status != "ok":
                        await _answer_free_empty(message)
                        return
                    claimed = True
                    item = _open_quiz_item(conn, user_id, state, spend=False)
                    if item is not None:
                        text = _format_quiz_item(item, int(state["q_index"]), len(bank))
            finally:
                conn.close()
        if item is None:
            await message.answer(
                f"Барлық сұрақ аяқталды. Ұпайың: {state['score']}.",
                reply_markup=_home_keyboard(),
            )
            return
        await message.answer(text, reply_markup=_home_keyboard())
    finally:
        if claimed:
            _release_section(user_id, "quiz")


async def _handle_quiz_answer(message: Message, user_id: str, user_text: str, api_key: str) -> None:
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            state = _quiz_row(conn, user_id)
        finally:
            conn.close()
    if not state["shown"] or state.get("current_id") is None:
        if int(state["q_index"]) == 0:
            await message.answer(
                _quiz_intro_for(user_id),
                reply_markup=_begin_keyboard(),
            )
            return
        await message.answer(
            "Келесі сұраққа өту үшін батырманы басыңыз.",
            reply_markup=_next_keyboard(),
        )
        return
    item = _bank_by_id().get(int(state["current_id"]))
    if item is None:
        _close_quiz_round(user_id, state)
        await message.answer(
            "Келесі сұраққа өту үшін батырманы басыңыз.",
            reply_markup=_next_keyboard(),
        )
        return
    bot = message.bot
    previous = str(state.get("first_try") or "") if int(state["attempts"]) >= 1 else ""
    try:
        if bot is not None:
            async with ChatActionSender(
                bot=bot,
                chat_id=message.chat.id,
                action=ChatAction.TYPING,
                interval=TYPING_INTERVAL,
                initial_sleep=0,
            ):
                grade = await _judge_answer(item, user_text, api_key, previous)
        else:
            grade = await _judge_answer(item, user_text, api_key, previous)
    except Exception:
        await message.answer(API_ERROR)
        return
    if grade is None:
        await message.answer(API_ERROR)
        return
    note = grade["explanation"] or _fallback_note(item)
    if grade["verdict"] == "correct":
        state["score"] = int(state["score"]) + 1
        _close_quiz_round(user_id, state)
        await message.answer(
            f"✅ Дұрыс! Ұпай: {state['score']}\n\n{note}",
            reply_markup=_next_keyboard(),
        )
        return
    if grade["verdict"] == "partial" and int(state["attempts"]) < 1:
        state["attempts"] = 1
        state["first_try"] = user_text
        state["mode"] = "quiz"
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                _save_progress(conn, user_id, state)
            finally:
                conn.close()
        await message.answer(
            _partial_line(grade["found"], grade["left"]),
            reply_markup=_home_keyboard(),
        )
        return
    if int(state["attempts"]) < 1:
        state["attempts"] = 1
        state["first_try"] = ""
        state["mode"] = "quiz"
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                _save_progress(conn, user_id, state)
            finally:
                conn.close()
        await message.answer(QUIZ_RETRY, reply_markup=_home_keyboard())
        return
    _close_quiz_round(user_id, state)
    await message.answer(
        _reveal_answer(str(item["answer"]), note),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_next_keyboard(),
    )


@router.message(F.photo)
async def on_receipt(message: Message) -> None:
    sender = message.from_user
    if sender is None or not message.photo or message.bot is None:
        return
    user_id = str(sender.id)
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            waiting = _awaiting_receipt(conn, user_id)
        finally:
            conn.close()
    if not waiting:
        return
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Растау", callback_data=f"pro:ok:{user_id}")],
        ]
    )
    try:
        await message.bot.send_photo(
            int(ADMIN_ID),
            message.photo[-1].file_id,
            caption=f"Төлем чегі\nTelegram id: {user_id}",
            reply_markup=markup,
        )
    except Exception:
        await message.answer(API_ERROR)
        return
    with _quiz_lock:
        conn = _quiz_connect()
        try:
            _set_await_receipt(conn, user_id, False)
        finally:
            conn.close()
    await message.answer("Чек жіберілді. Растауды күтіңіз.")


@router.message()
async def on_question(message: Message, api_key: str) -> None:
    question = (message.text or "").strip()
    if not question or question.startswith("/"):
        return
    sender = message.from_user
    user_id = str(sender.id) if sender is not None else ""
    if user_id:
        with _quiz_lock:
            conn = _quiz_connect()
            try:
                mode = _quiz_row(conn, user_id)["mode"]
            finally:
                conn.close()
        if mode == "quiz":
            await _handle_quiz_answer(message, user_id, question, api_key)
            return
        status = _claim_section(user_id, "ask")
        if status == "busy":
            return
        if status != "ok":
            await _answer_free_empty(message)
            return
    else:
        status = "ok"
    bot = message.bot
    if bot is None:
        if user_id:
            _release_section(user_id, "ask")
        await message.answer(API_ERROR)
        return
    try:
        pending = await message.answer(THINKING)
        # Telegram clears "typing" after about 5 seconds, so refresh it until the reply is sent.
        async with ChatActionSender(
            bot=bot,
            chat_id=message.chat.id,
            action=ChatAction.TYPING,
            interval=TYPING_INTERVAL,
            initial_sleep=0,
        ):
            try:
                if _open_client() is None:
                    await _deliver_answer(pending, MISSING_DB)
                    return
                await _deliver_answer(pending, await answer_question(question, api_key))
            except Exception:
                await _deliver_answer(pending, API_ERROR)
    finally:
        if user_id:
            _release_section(user_id, "ask")


_token, _api_key = _require_env()
dp1 = Dispatcher(api_key=_api_key)
dp1.include_router(router)
bot1 = Bot(token=_token)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    await dp1.start_polling(bot1)
