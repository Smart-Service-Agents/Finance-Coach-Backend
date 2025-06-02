import os
import json
import numpy as np
import faiss
from django.conf import settings


class FinanceCoach:
    def __init__(self):
        # Paths
        self.BASE_DIR      = settings.BASE_DIR
        self.dataset_path  = os.path.join(self.BASE_DIR, "chatbot", "dataset.json")
        # Gemini API key
        self.GEMINI        = os.getenv("Gemini_Key")

        # Lazy‐initialized attributes
        self.embedder      = None      # SentenceTransformer
        self.index         = None      # FAISS index
        self.transcripts   = None      # List of transcript strings
        self.video_links   = None      # List of corresponding video URLs
        self.dataset_loaded = False

    def _load_dataset_and_index(self):
        """
        Load JSON dataset of {transcript, video} pairs, build an embedding index.
        This runs only once on first call to answer_question().
        """
        if self.dataset_loaded:
            return

        # 1) Read dataset.json
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 2) Extract transcripts & video URLs
        self.transcripts = [entry["transcript"] for entry in raw_data]
        self.video_links = [entry["video"]      for entry in raw_data]

        # 3) Lazy‐import SentenceTransformer
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 4) Encode all transcripts to 384-dimensional vectors
        all_embeddings = self.embedder.encode(self.transcripts, convert_to_numpy=True).astype(np.float32)

        # 5) Build a FAISS index (Flat L2 on 384 dims)
        self.index = faiss.IndexFlatL2(384)
        self.index.add(all_embeddings)

        self.dataset_loaded = True

    def _find_best_match(self, question: str):
        """
        Return (best_transcript, best_video_url) by finding nearest neighbor in FAISS.
        """
        self._load_dataset_and_index()
        # 1) Embed question
        q_emb = self.embedder.encode(question, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        # 2) Search the single nearest neighbor
        _, idxs = self.index.search(q_emb, 1)
        best_idx = int(idxs[0][0])
        return self.transcripts[best_idx], self.video_links[best_idx]

    def _generate_with_gemini(self, prompt: str) -> str:
        """
        Lazy‐import and call Google Gemini to refine the answer.
        """
        import google.generativeai as genai
        genai.configure(api_key=self.GEMINI)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    def answer_question(self, question: str) -> dict:
        """
        1. Find best matching transcript + video link from dataset.
        2. Build a Gemini prompt that asks for a concise, hotel‐finance‐oriented answer.
        3. Return { "answer": <gemini_output>, "video": <video_url> }.
        """
        # 1) Find best transcript/video
        best_transcript, video_url = self._find_best_match(question)

        # 2) Construct a prompt for Gemini to turn transcript snippet into a proper response
        gemini_prompt = (
            "You are an expert hotel finance assistant.\n"
            f"Context excerpt:\n\"\"\"\n{best_transcript}\n\"\"\"\n\n"
            f"User question: \"{question}\"\n\n"
            "Provide a concise, professional answer based on the excerpt above. "
            "If the excerpt does not fully answer the question, combine its information "
            "with best practices in hotel finance and explain clearly."
        )

        # 3) Call Gemini
        try:
            refined_answer = self._generate_with_gemini(gemini_prompt)
        except Exception as e:
            # If Gemini call fails, fall back to returning the raw transcript
            refined_answer = (
                "Error refining answer with Gemini; here is the closest context:\n\n"
                f"{best_transcript}"
            )

        return {
            "answer": refined_answer,
            "video": video_url
        }
