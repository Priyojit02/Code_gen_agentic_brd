# utils/job_utils.py
"""
Document splitting utilities.

Detects:
  - "SECTION: 1. Purpose"
  - inline numeric headings like "5.1. Functional Requirement"
  - inline concatenations like "Structure Scope5.1. Functional Requirement"

Returns a dict:
  - keys for granular sections: "5.2 technical architecture" (title preserved)
  - numeric keys for parents: "5" contains merged text of 5.* (useful for get_section_text("5"))
  - "full" -> entire document
"""
import re
from typing import Dict

# Matches explicit SECTION: headers (captures the numeric part and the title)
SECTION_HEADER_RE = re.compile(
    r"SECTION:\s*(\d+(?:\.\d+)*)\.?\s*([^\n\r]*)",
    re.IGNORECASE | re.MULTILINE,
)

# Matches numeric headings like "5.1. Title" possibly at start of line
INLINE_NUMERIC_HEADER_RE = re.compile(
    r"(?m)^(?P<num>\d+(?:\.\d+)*)\.\s*(?P<title>[^\n\r]*)"
)

def _normalize_key(num: str, title: str) -> str:
    """
    Build a human-friendly key. Examples:
      num="5.2", title="Technical Architecture" -> "5.2 technical architecture"
      num="5", title="" -> "5"
    """
    if title:
        # normalize spacing and lowercase the title
        title_norm = re.sub(r"\s+", " ", title.strip()).lower()
        return f"{num} {title_norm}"
    return num

def split_sections(document: str) -> Dict[str, str]:
    if not document:
        return {"full": ""}

    # --- Preprocessing: insert newline before inline sub-section numbers that are
    # glued to previous text (e.g. "Scope5.1. ...") so they become line-start headers.
    # We look for sequences like "xxx5.1." where the digits are not preceded by a newline.
    # This is conservative: only insert newline where a numeric-dot sequence with at least one dot exists.
    preprocessed = re.sub(r"(?<=\D)(?=(\d+(?:\.\d+)+\.))", "\n", document)

    # We'll collect header positions (start index) and metadata (num, title)
    headers = []

    # 1) Find explicit SECTION: headers
    for m in SECTION_HEADER_RE.finditer(preprocessed):
        num = m.group(1).strip()
        title = m.group(2).strip()
        headers.append({"start": m.start(), "end": m.end(), "num": num, "title": title})

    # 2) Find inline numeric headers at line starts (after preprocessing)
    #    but avoid duplicating the SECTION: ones already captured (they will match too sometimes).
    for m in INLINE_NUMERIC_HEADER_RE.finditer(preprocessed):
        # If this numeric header is already represented by a SECTION: match at same position, skip it
        if any(abs(m.start() - h["start"]) <= 2 for h in headers):
            continue
        num = m.group("num").strip()
        title = m.group("title").strip()
        headers.append({"start": m.start(), "end": m.end(), "num": num, "title": title})

    # If we found no headers at all, return full doc
    if not headers:
        return {"full": document.strip()}

    # Sort headers by position
    headers.sort(key=lambda h: h["start"])

    sections = {}
    for idx, h in enumerate(headers):
        start = headers[idx]["end"]
        end = headers[idx + 1]["start"] if idx + 1 < len(headers) else len(preprocessed)
        num = headers[idx]["num"]
        title = headers[idx]["title"]
        key = _normalize_key(num, title)
        content = preprocessed[start:end].strip()
        sections[key] = content

    # Build merged parent-level sections. E.g., "5" contains everything of "5", "5.1", "5.2", ...
    merged_parents: Dict[str, str] = {}
    for key, content in sections.items():
        # numeric prefix is the leading number(s), e.g., "5" from "5.2 ..." or "5" if key == "5"
        num_prefix = key.split(" ", 1)[0]  # "5.2" or "5"
        parent_root = num_prefix.split(".")[0]  # "5"
        merged_parents.setdefault(parent_root, "")
        merged_parents[parent_root] += ("\n\n" + f"--- [{key}] ---\n" + content).strip()

    # Merge parent sections back (don't overwrite existing granular keys)
    # Parent numeric key (like "5") will contain merged content; if a granular "5" key existed keep it appended
    for parent_key, merged_content in merged_parents.items():
        existing = sections.get(parent_key, "")
        if existing:
            sections[parent_key] = (existing + "\n\n" + merged_content).strip()
        else:
            sections[parent_key] = merged_content.strip()

    # Always include full fallback
    sections["full"] = document.strip()

    return sections
