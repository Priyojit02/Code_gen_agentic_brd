import re
from typing import Dict

def split_sections(document: str) -> Dict[str, str]:
    """
    Split text by SECTION markers like 'SECTION: 1.', 'SECTION: 2.', etc.
    Returns a dict: { '1': <content>, '2': <content>, ... }
    """
    if not document:
        return {}

    # Normalize to avoid weird encodings or spacing
    doc = re.sub(r"[^\x00-\x7F]+", " ", document)
    doc = doc.replace("\r", "").strip()

    # Regex to capture SECTION headers
    pattern = re.compile(r"SECTION\s*:\s*(\d+(?:\.\d+)*)\s*\.?", re.IGNORECASE)
    matches = list(pattern.finditer(doc))
    sections = {}

    for i, match in enumerate(matches):
        section_num = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        content = doc[start:end].strip()
        sections[section_num] = content

    # Debug print
    print("\n===== PARSED SECTIONS =====")
    for num, text in sections.items():
        preview = text[:300].replace("\n", " ")
        print(f"\n🔹 SECTION {num}\nCONTENT PREVIEW:\n{preview}{'...' if len(text) > 300 else ''}")
        print("-" * 60)

    return sections