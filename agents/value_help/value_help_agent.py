"""
value_help_agent.py

Generates RAP-ready Value Help CDS views (F4 help) using LLM
and optional RAG knowledge base.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent

load_dotenv()

# Disable LangSmith tracing unless key is provided
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class ValueHelpAgent(BaseAgent):
    """
    Generates RAP-ready Value Help CDS views (F4 help) using LLM
    and optional RAG knowledge base.
    """

    # ------------------- LLM INIT -------------------
    def _init_llm(self):
        """Initialize the LLM client."""
        return ChatOpenAI(
            model_name=os.getenv("VALUE_HELP_MODEL_NAME", "gpt-4.1"),
            temperature=float(os.getenv("VALUE_HELP_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ------------------- VECTORSTORE INIT -------------------
    def _init_vectorstore(self):
        """Load or build FAISS vector DB from value_help_RAG_KB.txt."""
        kb_path = Path(__file__).parent / "value_help_RAG_KB.txt"
        vs_path = Path(__file__).parent / "value_help_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Reuse existing vectorstore if KB not updated
        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max((f.stat().st_mtime for f in vs_path.glob("**/*")), default=0)

            if kb_mtime <= vs_mtime:
                self.logger.info("📂 Loading existing FAISS vector DB for ValueHelpAgent...")
                return FAISS.load_local(
                    vs_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                self.logger.info("♻ KB updated — rebuilding FAISS index...")

        if not kb_path.exists():
            self.logger.warning(f"⚠ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)

        self.logger.info("✅ New FAISS vector DB created for ValueHelpAgent.")
        return vectorstore

    # ------------------- RETRIEVE CONTEXT -------------------
    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve top-k relevant KB chunks."""
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()

        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)

        if not results:
            return ""

        combined = "\n\n".join([r.page_content for r in results])

        self.logger.info(f"📚 Retrieved {len(results)} RAG context chunks for Value Help.")
        return combined

    # ------------------- SAFE JSON PARSER -------------------
    @staticmethod
    def _safe_parse_json(raw_text: str) -> dict:
        """
        Extracts and parses JSON from LLM output, fixing invalid
        backslash escapes that the LLM puts inside CDS code strings.
        """

        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        # Find outermost JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")

        json_str = match.group()

        # Attempt 1: parse as-is
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Attempt 2: fix invalid backslashes
        fixed = re.sub(
            r'(?<!\\)\\(?!["\\/bfnrtu])',
            r"\\\\",
            json_str,
        )

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 3: extract manually via regex
        code_match = re.search(
            r'"value_help_code"\s*:\s*"(.*?)"\s*[},]',
            cleaned,
            re.DOTALL,
        )

        purpose_match = re.search(
            r'"value_help_purpose"\s*:\s*"(.*?)"\s*[},]',
            cleaned,
            re.DOTALL,
        )

        if code_match:
            return {
                "value_help_code": code_match.group(1).replace("\\n", "\n"),
                "value_help_purpose": purpose_match.group(1) if purpose_match else "",
            }

        raise ValueError("Could not parse JSON from LLM output after all recovery attempts.")

    # ------------------- MAIN RUN -------------------
    def run(self, field_description: str, metadata=None):
        """
        Generate a RAP Value Help CDS view definition for a standard field.
        """

        if not field_description:
            self.logger.warning("No field description provided to ValueHelpAgent.")
            return {"type": "value_help", "purpose": "", "code": ""}

        self.logger.info("🚀 Running ValueHelpAgent with provided field description.")

        # Retrieve RAG context
        rag_context = self._get_relevant_context(field_description)

        full_context = field_description.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # System prompt
        system_message = SystemMessage(
            content=(
                "You are an SAP ABAP expert specializing in the RAP model. "
                "Generate valid CDS view entities for Value Help (F4 help). "
                "Use retrieved RAG context for reference."
            )
        )

        user_prompt = f"""
Field requirement:
{field_description.strip()}

Reference knowledge (from RAG KB if available):
{rag_context}

Output strictly in JSON format with two keys:
1) "value_help_code": RAP-ready ABAP CDS view entity code for value help.
2) "value_help_purpose": Short description of the value help CDS view purpose.
"""

        # LLM generation
        resp = self.llm.invoke([system_message, HumanMessage(content=user_prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # Parse JSON safely
        try:
            data = self._safe_parse_json(raw)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        # Normalize parsed data
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                data_obj = data[0]
            else:
                self.logger.warning("⚠ LLM returned a list without dict items; coercing to empty dict.")
                data_obj = {}
        elif isinstance(data, dict):
            data_obj = data
        else:
            self.logger.warning(f"⚠ Unexpected LLM output type: {type(data)}; coercing to empty dict.")
            data_obj = {}

        # Extract fields safely
        value_help_code = str(data_obj.get("value_help_code", "")).strip()
        value_help_purpose = str(data_obj.get("value_help_purpose", "")).strip()
        value_help_entity = str(data_obj.get("value_help_entity", "")).strip()

        self.logger.info(f"📄 Purpose: {value_help_purpose}")

        return {
            "type": "value_help",
            "purpose": value_help_purpose,
            "code": value_help_code,
            "entity": value_help_entity,
        }