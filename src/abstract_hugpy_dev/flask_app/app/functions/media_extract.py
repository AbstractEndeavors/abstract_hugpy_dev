"""Deterministic media-ingest helpers for the `/ml` amenities that are NOT model
inference — reading text out of an uploaded document (PDF / DOCX / plain text).

These are the EXTRACT stage of the media-intelligence pipeline (extract → enrich:
summarize + keywords), so they live as thin LOCAL handlers rather than riding the
model resolver: there is no model to load, nothing to delegate to a GPU worker. The
heavy parsers are imported lazily so importing this module stays cheap (no torch,
phone-clean) and a missing optional parser degrades to a clear error, not an
ImportError at module load.

Security: reads are JAILED to the storage root (UPLOADS_HOME / DEFAULT_ROOT). The
client only ever passes paths produced by POST /uploads (under UPLOADS_HOME), so a
path that resolves outside the root is rejected — this endpoint never becomes an
arbitrary-file-read primitive.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse, urljoin

from abstract_hugpy_dev.imports.src.constants.constants import UPLOADS_HOME, DEFAULT_ROOT

_MAX_URL_BYTES = 5 * 1024 * 1024  # cap fetched body so a huge/streamed page can't OOM us
_URL_TIMEOUT = 15                 # per-request seconds
_MAX_REDIRECTS = 4

_MAX_ASSESS_CHARS = 12000   # token budget for the structured assessment's body text
_MAX_ASSESS_LINKS = 50      # cap same-domain links carried in an assessment

# Extensions handled by each strategy. Anything else → a clear "unsupported" error.
_PDF_EXT = {".pdf"}
_DOCX_EXT = {".docx"}
_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".log", ".text"}


def _jailed_realpath(path: str) -> str:
    """Resolve ``path`` and require it to live under the storage root.

    Raises PermissionError if it escapes (symlinks resolved via realpath), so a
    crafted path like ``../../etc/passwd`` can't be read through this endpoint.
    """
    rp = os.path.realpath(path)
    roots = [os.path.realpath(r) for r in (UPLOADS_HOME, DEFAULT_ROOT) if r]
    if not any(rp == root or rp.startswith(root + os.sep) for root in roots):
        raise PermissionError("file path is outside the allowed storage root")
    return rp


def _extract_pdf(path: str) -> tuple[list[dict], str]:
    """Per-page text via pdfplumber, falling back to PyPDF2 when a page yields
    nothing (scanned/odd PDFs). No OCR — that's the heavy [ocr] path, not this one."""
    pages: list[dict] = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, pg in enumerate(pdf.pages):
                pages.append({"index": i, "text": (pg.extract_text() or "").strip()})
    except Exception:
        pages = []
    if not any(p["text"] for p in pages):
        # pdfplumber found nothing (or errored) — try PyPDF2 before giving up.
        try:
            import PyPDF2
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                pages = [{"index": i, "text": (pg.extract_text() or "").strip()}
                         for i, pg in enumerate(reader.pages)]
        except Exception:
            pass
    text = "\n\n".join(p["text"] for p in pages if p["text"]).strip()
    return pages, text


def _extract_docx(path: str) -> tuple[list[dict], str]:
    import docx  # python-docx
    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs if p.text and p.text.strip()).strip()
    return [], text


def _extract_text(path: str) -> tuple[list[dict], str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [], fh.read().strip()


_MAX_TEXT_BYTES = 16 * 1024 * 1024  # cap a generic read so a huge binary can't OOM us


def _extract_generic_text(path: str) -> tuple[list[dict], str]:
    """Reader of last resort: read an UNKNOWN file as UTF-8 text, but only when it
    actually looks textual.

    Mirrors the default in abstract_utilities' read_any_file ("if nothing else
    matches, just read it") — kept stdlib-only here (no pandas/ocr). A cheap
    binary sniff (a NUL byte, or too high a ratio of replacement/control chars)
    declines binary files (images, archives, spreadsheets) instead of handing
    back replacement-character garbage.
    """
    with open(path, "rb") as fh:
        raw = fh.read(_MAX_TEXT_BYTES)
    if not raw or b"\x00" in raw:
        return [], ""
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return [], ""
    bad = sum(1 for ch in text if ch == "�" or (ord(ch) < 32 and ch not in "\t\n\r\f"))
    if bad / len(text) > 0.10:
        return [], ""
    return [], text.strip()


def extract_document(path: str) -> dict:
    """Read text from a document under the storage root.

    Returns the same {ok, text, ...} shape the other `/ml` amenities produce, so
    the chat's pickText() narration path treats it identically. ``pages`` is
    populated for PDFs (per-page), empty otherwise.
    """
    rp = _jailed_realpath(path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"no such file: {os.path.basename(path)}")

    ext = os.path.splitext(rp)[1].lower()
    if ext in _PDF_EXT:
        pages, text = _extract_pdf(rp)
        kind = "pdf"
    elif ext in _DOCX_EXT:
        pages, text = _extract_docx(rp)
        kind = "docx"
    elif ext in _TEXT_EXT:
        pages, text = _extract_text(rp)
        kind = "text"
    else:
        # Unknown extension → default to "just read it" (guarded UTF-8 read),
        # so an unrecognized file is read rather than rejected outright.
        pages, text = _extract_generic_text(rp)
        kind = "text"
        if not text:
            return {"ok": False, "kind": "unknown", "text": "",
                    "error": f"file type '{ext or '(none)'}' isn't readable as text"}

    if not text:
        return {"ok": False, "kind": kind, "text": "", "pages": pages,
                "error": "no extractable text (the document may be empty or image-only/scanned)"}

    return {
        "ok": True,
        "kind": kind,
        "text": text,
        "pages": pages,
        "chars": len(text),
        "name": os.path.basename(path),
    }


# --- URL fetch (readable text from a webpage) --------------------------------
# Server-side URL fetch is an SSRF surface: a caller could try to make the server
# hit internal services (169.254.169.254 cloud metadata, localhost:7002, LAN
# workers, etc.). We defend by resolving every host (including each redirect hop)
# and REFUSING any address that isn't public — scheme is restricted to http(s),
# redirects are followed manually so each Location is re-validated, and the body
# is size-capped.

def _assert_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise PermissionError(f"only http/https URLs are allowed (got '{p.scheme or 'none'}')")
    host = p.hostname
    if not host:
        raise PermissionError("URL has no host")
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise PermissionError(f"cannot resolve host '{host}'")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise PermissionError(f"refusing to fetch a non-public address ({ip})")


def fetch_url_text(url: str) -> dict:
    """Fetch a public webpage and return its readable text (SSRF-guarded).

    Same {ok, text, ...} shape as extract_document. Strips script/style/nav noise
    via BeautifulSoup's built-in parser (no lxml dependency).
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "kind": "url", "text": "", "error": "no url provided"}
    if "://" not in url:
        url = "https://" + url  # bare domain → https

    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    current = url
    for _ in range(_MAX_REDIRECTS):
        _assert_public_url(current)  # re-validate EVERY hop (defeats redirect-based SSRF)
        resp = session.get(
            current, timeout=_URL_TIMEOUT, allow_redirects=False, stream=True,
            headers={"User-Agent": "hugpy-media-intelligence/1.0"},
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                break
            current = urljoin(current, loc)
            continue
        resp.raise_for_status()
        chunks, total = [], 0
        for chunk in resp.iter_content(8192):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_URL_BYTES:
                break
        html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        text = "\n".join(ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip())
        if not text:
            return {"ok": False, "kind": "url", "url": current, "text": "",
                    "error": "no readable text found at that URL"}
        return {"ok": True, "kind": "url", "url": current, "title": title,
                "text": text, "chars": len(text)}
    return {"ok": False, "kind": "url", "text": "", "error": "too many redirects"}


def assess_url(url: str) -> dict:
    """Structured, LLM-ready webpage assessment, upgrading the plain URL read.

    Delegates to abstract_webtools' assessManager.assess_webpage, which fetches
    cheaply (requests) and only escalates to a full browser render when the page
    comes back JS-walled / near-empty — returning title, meta-description, JSON-LD,
    same-domain links and a token-budgeted body alongside the readable text.

    Posture (consistent with the rest of this module):
      * SSRF — the initial URL is validated at the front door (http/https only,
        public addresses) exactly like fetch_url_text. NOTE: assessManager follows
        redirects internally, so individual redirect HOPS are NOT re-validated the
        way fetch_url_text does — the front-door check is the guarantee here.
      * Cost — abstract_webtools (Selenium / matplotlib) is imported lazily on the
        first call only; a forced browser render happens only as an auto-fallback
        for a near-empty page, never by default.
      * Graceful — if abstract_webtools is absent, errors, or yields no text, this
        degrades to the lightweight requests+bs4 fetch_url_text(), so the amenity
        is never worse than a plain read. Same {ok, kind, text, ...} contract.
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "kind": "url", "text": "", "error": "no url provided"}
    if "://" not in url:
        url = "https://" + url  # bare domain → https
    _assert_public_url(url)  # refuse internal/loopback before any fetch (raises PermissionError)

    try:
        from abstract_webtools.managers.assessManager import assess_webpage as _aw_assess
    except Exception:
        return fetch_url_text(url)  # abstract_webtools not installed → lightweight read

    try:
        page = _aw_assess(url, max_chars=_MAX_ASSESS_CHARS, max_links=_MAX_ASSESS_LINKS)
    except Exception:
        return fetch_url_text(url)  # assessment blew up → lightweight read

    text = (page.get("text") or "").strip() if isinstance(page, dict) else ""
    if not text:
        # assessment produced no usable body (render failed / empty) — plain reader.
        return fetch_url_text(url)

    return {
        "ok": True,
        "kind": "url",
        "url": page.get("url") or url,
        "title": page.get("title"),
        "description": page.get("description"),
        "text": text,
        "metadata": page.get("metadata") or [],
        "jsonld": page.get("jsonld") or [],
        "links": page.get("links") or [],
        "truncated": bool(page.get("truncated")),
        "render": page.get("render"),
        "chars": len(text),
    }
