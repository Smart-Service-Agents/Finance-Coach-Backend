from django.conf import settings
import os
import json
import numpy as np
import faiss

class FinanceCoach:
    def __init__(self):
        self.BASE_DIR     = settings.BASE_DIR
        self.dataset_path = os.path.join(self.BASE_DIR, "chatbot", "dataset.json")
        self.GEMINI       = os.getenv("Gemini_Key")
        self.embedder     = None
        self.index        = None
        self.transcripts  = None
        self.video_links  = None
        self.dataset_loaded = False

    def _load_dataset_and_index(self):
        if self.dataset_loaded:
            return

        # 1) Read JSON file
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 2) Extract transcripts & video URLs
        self.transcripts = [e["transcript"] for e in raw_data]
        self.video_links = [e["video"]      for e in raw_data]

        # 3) Lazy import a very small MiniLM (≈38 MB)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # 4) Encode entire dataset at once (N×384 floats)
        all_embeddings = self.embedder.encode(
            self.transcripts,
            convert_to_numpy=True
        ).astype(np.float32)

        # 5) Build a FAISS Flat L2 index on 384-dim vectors
        self.index = faiss.IndexFlatL2(384)
        self.index.add(all_embeddings)
        self.dataset_loaded = True

    def _find_best_match(self, question: str):
        self._load_dataset_and_index()
        q_emb = self.embedder.encode(question, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        _, idxs = self.index.search(q_emb, 1)
        best_idx = int(idxs[0][0])
        return self.transcripts[best_idx], self.video_links[best_idx]

    def _generate_with_gemini(self, prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.GEMINI)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text

    def answer_question(self, question: str) -> dict:
        best_transcript, video_url = self._find_best_match(question)

        gemini_prompt = (
            "You are an expert hotel finance assistant.\n"
            f"Context excerpt:\n\"\"\"\n{best_transcript}\n\"\"\"\n\n"
            f"User question: \"{question}\"\n\n"
            "Provide a concise, professional answer based on the excerpt. "
            "If the excerpt alone isn’t enough, combine with hotel‐finance best practices."
        )

        try:
            refined = self._generate_with_gemini(gemini_prompt)
        except Exception:
            refined = (
                "Could not reach Gemini. Here’s the closest context:\n\n"
                f"{best_transcript}"
            )

        return {"answer": refined, "video": video_url}
