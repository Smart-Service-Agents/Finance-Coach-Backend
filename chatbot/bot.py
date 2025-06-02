import os
import json
import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FinanceCoach:
    def __init__(self):
        # Paths
        self.BASE_DIR      = settings.BASE_DIR
        self.dataset_path  = os.path.join(self.BASE_DIR, "chatbot", "dataset.json")
        # Gemini API key
        self.GEMINI        = os.getenv("Gemini_Key")

        # Lazy‐initialized attributes
        self.vectorizer      = None    # TfidfVectorizer
        self.tfidf_matrix    = None    # sparse matrix (N × features)
        self.transcripts     = None    # list of transcript strings
        self.video_links     = None    # list of corresponding video URLs
        self.dataset_loaded  = False

    def _load_dataset_and_index(self):
        """
        Load JSON dataset of { transcript, video } pairs, then build a TF-IDF index.
        This only runs once on first call to answer_question().
        """
        if self.dataset_loaded:
            return

        # 1) Read dataset.json
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 2) Extract transcripts & video URLs
        self.transcripts = [ entry["transcript"] for entry in raw_data ]
        self.video_links = [ entry["video"]      for entry in raw_data ]

        # 3) Build a TF-IDF vectorizer on all transcripts
        self.vectorizer   = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000
        )
        # 4) Fit & transform to get an N×F sparse matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.transcripts)

        self.dataset_loaded = True

    def _find_best_match(self, question: str):
        """
        Return (best_transcript, best_video_url) by computing cosine similarity
        between the question and all transcripts in TF-IDF space.
        """
        self._load_dataset_and_index()

        # 1) Transform the incoming question into the same TF-IDF space
        q_vec = self.vectorizer.transform([question])  # shape (1×F)

        # 2) Compute cosine similarities (N-dim array)
        sims = cosine_similarity(self.tfidf_matrix, q_vec).flatten()

        # 3) Identify the index of the highest similarity
        best_idx = int(np.argmax(sims))

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
        1. Find best matching transcript + video link using TF-IDF & cosine similarity.
        2. Build a Gemini prompt that asks for a concise, hotel‐finance‐oriented answer.
        3. Return { "answer": <gemini_output>, "video": <video_url> }.
        """
        # 1) Find the best transcript/video
        best_transcript, video_url = self._find_best_match(question)

        # 2) Construct a prompt for Gemini to turn the transcript snippet into a polished response
        gemini_prompt = (
            "You are an expert hotel finance assistant.\n"
            f"Context excerpt:\n\"\"\"\n{best_transcript}\n\"\"\"\n\n"
            f"User question: \"{question}\"\n\n"
            "Provide a concise, professional answer based on the excerpt above. "
            "If the excerpt alone doesn’t fully answer, combine its information "
            "with standard hotel finance best practices and explain clearly."
        )

        # 3) Call Gemini; fall back to raw excerpt if it fails
        try:
            refined_answer = self._generate_with_gemini(gemini_prompt)
        except Exception:
            refined_answer = (
                "Could not refine via Gemini. Here’s the closest context:\n\n"
                f"{best_transcript}"
            )

        return {
            "answer": refined_answer,
            "video":  video_url
        }
