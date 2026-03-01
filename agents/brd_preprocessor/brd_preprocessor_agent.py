"""
brd_preprocessor_agent.py

Converts raw BRD (Business Requirement Document) text into the
fixed 9-section format expected by the downstream agent pipeline.
"""

import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()

# — Fixed section template —
REQUIRED_SECTIONS = [
    ("1", "Purpose"),
    ("2", "Scope"),
    ("3", "Structure"),
    ("4", "Table"),
    ("5", "Value Help"),
    ("6", "CDS"),
    ("7", "Function Module"),
    ("8", "Global Class"),
    ("9", "Report Program"),
]


class BrdPreprocessorAgent(BaseAgent):
    """
    Reads a raw BRD and converts it into the standardised
    SECTION: N. format that split_sections() and all downstream
    agents expect. Every section heading is guaranteed present.
    """

    # — LLM initialisation (same pattern as other agents) —
    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("BRD_PREPROCESSOR_MODEL_NAME", "gpt-4.1"),
            temperature=float(os.getenv("BRD_PREPROCESSOR_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # — Safety-net: guarantee all 9 headings exist —
    @staticmethod
    def ensure_all_sections(text: str) -> str:
        """
        Scans the LLM output and injects any missing SECTION headings
        with 'N/A' content so the downstream pipeline never breaks.
        """
        for num, title in REQUIRED_SECTIONS:
            pattern = rf"SECTION:\s*{num}\."
            if not re.search(pattern, text, re.IGNORECASE):
                text += f"\n\nSECTION: {num}. {title}\n\nN/A"
                logging.getLogger("BrdPreprocessorAgent").warning(
                    f"Section {num} ({title}) was missing — injected with N/A."
                )
        return text.strip()

    # — Main execution —
    def run(self, brd_text: str, metadata=None) -> str:
        """
        Accepts raw BRD text, returns a formatted string with all
        9 SECTION: N. headings populated (or marked N/A).
        """

        if not brd_text or not brd_text.strip():
            self.logger.warning("Empty BRD text received — returning skeleton.")
            return self._build_skeleton()

        self.logger.info("Running BrdPreprocessorAgent — converting BRD")

        # — System message —
        system_message = SystemMessage(
            content=(
                "You are an expert SAP technical architect. Your job is to read a "
                "Business Requirement Document (BRD) and restructure its content "
                "into a fixed SAP Technical Specification format with exactly 9 sections. "
                "You must preserve ALL technical details, field names, table names, "
                "data elements, method signatures, and any other specific information. "
                "Do NOT summarise or lose information — redistribute it correctly."
            )
        )

        # — Section classification guide —
        section_guide = """
SECTION: 1. Purpose
→ Business objective, why this development is needed, problem statement.

SECTION: 2. Scope
→ What type of SAP object/program is being built (report, interface, enhancement, etc.).

SECTION: 3. Structure
→ ABAP DDIC structures, type definitions, field lists used for internal data.
→ Include structure name, field names, data elements, and descriptions if available.

SECTION: 4. Table
→ Custom transparent/cluster/pool database tables, field catalogs with data elements.
→ Include table name, field names, data elements, key indicators, and descriptions.

SECTION: 5. Value Help
→ Search helps, F4 helps, value help CDS views, dropdown value providers.
→ Include field name the value help applies to, source table/view, and any filters.

SECTION: 6. CDS
→ CDS View entities, associations, joins between tables, calculated fields.
→ NOT value help CDS (those go in Section 5).
→ Include view name, source tables, fields to fetch, key fields, and associations.

SECTION: 7. Function Module
→ Function modules, import/export/changing/tables parameters, exceptions, RFC settings.
→ Include FM name, parameter names with types, logic description, and exceptions.

SECTION: 8. Global Class
→ ABAP OO global classes (SE24/Eclipse), interfaces, methods, attributes, constructors.
→ Include class name, method signatures (IMPORTING/EXPORTING/RETURNING/CHANGING).

SECTION: 9. Report Program
→ ABAP report programs, selection screens (parameters, select-options), ALV.
→ Include report name, selection screen fields, ALV columns, processing steps.
"""

        # — Prompt —
        prompt = f"""
Below is a Business Requirement Document (BRD). Read it carefully and redistribute
ALL of its content into exactly 9 sections using the format shown below.

RULES:
1. You MUST output ALL 9 section headings exactly as: "SECTION: N. Title"
2. If the BRD has NO information for a section, write the heading followed by N/A.
3. NEVER skip or omit any section heading — all 9 must appear.
4. Preserve ALL technical details — field names, table names, data elements, etc.
5. Do NOT add information that is not in the BRD. Only redistribute what is given.
6. Do NOT wrap output in markdown, JSON, or code blocks. Output plain text only.
7. Start your output with "SAP Technical Specification Document" on the first line.

SECTION CLASSIFICATION GUIDE:
{section_guide}

BRD CONTENT:
----------------
{brd_text}
----------------

OUTPUT FORMAT:
SAP Technical Specification Document

SECTION: 1. Purpose
<content or N/A>

SECTION: 2. Scope
<content or N/A>

SECTION: 3. Structure
<content or N/A>

SECTION: 4. Table
<content or N/A>

SECTION: 5. Value Help
<content or N/A>

SECTION: 6. CDS
<content or N/A>

SECTION: 7. Function Module
<content or N/A>

SECTION: 8. Global Class
<content or N/A>

SECTION: 9. Report Program
<content or N/A>
"""

        # — Call LLM —
        resp = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # — Strip markdown fences if LLM wraps output —
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

        self.logger.info(f"BrdPreprocessorAgent LLM output length: {len(raw)}")

        # — Layer 2 safety net —
        formatted_text = self.ensure_all_sections(raw)

        self.logger.info("BrdPreprocessorAgent completed — all 9 sections ensured")
        return formatted_text

    # — Skeleton builder (fallback for empty input) —
    def _build_skeleton(self) -> str:
        """Returns a skeleton with all 9 section headings and N/A content."""
        lines = ["SAP Technical Specification Document"]
        for num, title in REQUIRED_SECTIONS:
            lines.append(f"\nSECTION: {num}. {title}")
            lines.append("N/A")
        return "\n".join(lines)