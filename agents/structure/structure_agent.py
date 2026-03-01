import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class StructureAgent(BaseAgent):
    """
    Generates an ABAP DDIC structure with purpose using RAG context from a vector database.
    """

    # --- LLM Setup ---
    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("STRUCTURE_MODEL_NAME", "gpt-4.1"),
            temperature=float(os.getenv("STRUCTURE_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # --- Vector DB Setup ---
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from the Structure_RAG_KB.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "Structure_RAG_KB.txt"
        vs_path = Path(os.path.dirname(__file__)) / "structure_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Rebuild vectorstore if KB updated after FAISS creation
        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max(f.stat().st_mtime for f in vs_path.glob("**/*"))
            if kb_mtime <= vs_mtime:
                self.logger.info("📚 Loading existing FAISS vector DB for StructureAgent...")
                return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
            else:
                self.logger.info("🔄 KB updated — rebuilding FAISS index...")

        if not kb_path.exists():
            self.logger.warning(f"⚠️ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for StructureAgent.")
        return vectorstore
    
    # --- Retrieve relevant RAG chunks ---
    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks for StructureAgent.")
        return combined

    # --- Main Execution ---
    def run(self, section_text: str, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to StructureAgent.")
            return {"purpose": "", "path": Path(), "type": "structure"}

        self.logger.info("Running StructureAgent with provided section text...")
        print("\n🧩 [StructureAgent] Received text:\n", section_text[:500], "\n--- END ---")

        # --- Retrieve RAG context ---
        rag_context = self._get_relevant_context(section_text)
        if rag_context:
            section_text += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # --- System message ---
        system_message = SystemMessage(
            content=(
                "You are a senior SAP ABAP developer specializing in Data Dictionary (DDIC) design. "
                "Follow SAP field naming standards, domain/data element conventions, and maintain readability. "
                "Ensure structures are semantically meaningful and consistent with business context."
                "Do not generate invalid annotations, stick to RAG file"
            )
        )

        # --- Prompt for LLM ---
        prompt = f"""
        You are an expert SAP ABAP developer. Using the following requirement/context,
        produce and return JSON with two keys:
        1) "structure_code":  An ABAP DDIC structure DDL code which can be directly pasted
           in Eclipse to generate the structure with field names, data types, and short comments.
        2) "structure_purpose": A short 'PURPOSE' explaining the structure name and use of each field.

        Context:
        {section_text}

        Output format (strict JSON only, no markdown, no commentary):
        {{
            "structure_code": "...",
            "structure_purpose": "..."
        }}
        """

        # --- Call LLM ---
        resp = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # --- Parse JSON output ---
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        structure_code = data.get("structure_code", "").strip()
        structure_purpose = data.get("structure_purpose", "").strip()

        # --- Save structure code ---
        # out_path = self.job_dir / "Structure.txt"
        # out_path.write_text(structure_code, encoding="utf-8")

        self.logger.info(f"✅ StructureAgent output: {structure_code}")

        return {
            "purpose": structure_purpose,
            "code": structure_code,
            # "path": out_path,
            "type": "structure"
        }
