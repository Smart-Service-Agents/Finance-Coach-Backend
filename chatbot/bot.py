import os
import json
from django.conf import settings


class FinanceCoach:
    def __init__(self):
        self.BASE_DIR = settings.BASE_DIR
        self.model_path = os.path.join(self.BASE_DIR, "chatbot", "hotel_llm")
        self.dataset_path = os.path.join(self.BASE_DIR, "chatbot", "dataset.json")

        self.GEMINI = os.getenv("Gemini_Key")

        print(self.GEMINI)

        # All heavy components start as None and load on demand
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.embedder = None
        self.index = None
        self.dataset = None
        self.video_links = None

    def _load_generator(self):
        if self.generator is None:
            # Import here so it doesn’t block startup
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )

    def _load_embedder(self):
        if self.embedder is None:
            # Import here for SentenceTransformer
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_dataset(self):
        if self.dataset is None:
            # Lazy load dataset JSON
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                self.dataset = json.load(file)

            # Now load the embedder and build FAISS index
            self._load_embedder()

            # Import numpy and faiss on demand
            import numpy as np
            import faiss

            # Create a flat L2 index for 384-dimensional embeddings
            self.index = faiss.IndexFlatL2(384)
            self.video_links = []

            for entry in self.dataset:
                text_embedding = self.embedder.encode(entry["transcript"]).astype(np.float32)
                self.index.add(text_embedding.reshape(1, -1))
                self.video_links.append(entry["video"])

    def retrieve_best_transcript(self, question):
        # Ensure the dataset, embedder, and index are loaded
        self._load_dataset()
        import numpy as np

        question_embedding = self.embedder.encode(question).astype(np.float32).reshape(1, -1)
        _, best_match = self.index.search(question_embedding, 1)
        best_index = int(best_match[0][0])
        return self.dataset[best_index]["transcript"], self.video_links[best_index]

    def generate_prompt(self, prompt):
        # Lazy import and configure Google Generative AI
        import google.generativeai as genai

        genai.configure(api_key=self.GEMINI)
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(prompt)
        return response.text

    def answer_question(self, question):
        # Ensure the generator (transformers pipeline) is loaded
        self._load_generator()

        # Retrieve context from dataset via FAISS
        best_transcript, video_link = self.retrieve_best_transcript(question)
        prompt_for_transformer = f"Context: {best_transcript} Question: {question} Answer:"

        # Run the local seq2seq model
        result = self.generator(prompt_for_transformer, max_length=100)
        chunk = result[0]["generated_text"]

        # Build a second prompt for Gemini
        final_prompt = (
            f"You are an expert hotel finance analyst. "
            f"Analyze the question: {question}\n\n"
            f"Use this chunk from your database: {chunk}"
        )

        # Get the final answer via Google Generative AI
        answer = self.generate_prompt(final_prompt)

        return {"answer": answer, "video": video_link}
