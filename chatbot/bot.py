import os
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import google.generativeai as genai


class FinanceCoach:
    def __init__(self):
        self.BASE_DIR = settings.BASE_DIR
        self.model_path = os.path.join(self.BASE_DIR, "chatbot", "hotel_llm")
        self.dataset_path = os.path.join(self.BASE_DIR, "chatbot", "dataset.json")


        self.GEMINI = os.getenv("Gemini_Key")

        print(self.GEMINI)

        self.tokenizer = None
        self.model = None
        self.generator = None
        self.embedder = None
        self.dataset = None
        self.index = None
        self.video_links = None

    def _load_generator(self):
        if self.generator is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def _load_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_dataset(self):
        if self.dataset is None:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                self.dataset = json.load(file)

            self._load_embedder()
            self.index = faiss.IndexFlatL2(384)
            self.video_links = []

            for entry in self.dataset:
                text_embedding = self.embedder.encode(entry["transcript"]).astype(np.float32)
                self.index.add(text_embedding.reshape(1, -1))
                self.video_links.append(entry["video"])

    def retrieve_best_transcript(self, question):
        self._load_dataset()
        question_embedding = self.embedder.encode(question).astype(np.float32).reshape(1, -1)
        _, best_match = self.index.search(question_embedding, 1)
        best_index = int(best_match[0][0])
        return self.dataset[best_index]["transcript"], self.video_links[best_index]

    def generate_prompt(self, prompt):
        genai.configure(api_key=self.GEMINI)
        model = genai.GenerativeModel('gemini-2.0-flash')

        response = model.generate_content(prompt)

        return response.text

    def answer_question(self, question):
        self._load_generator()
        best_transcript, video_link = self.retrieve_best_transcript(question)
        prompt = f"Context: {best_transcript} Question: {question} Answer:"
        
        result = self.generator(prompt, max_length=100)
        chunk = result[0]['generated_text']

        prompt = f"Using the following transcript excerpts, answer the question: {question}\n\n {chunk}"

        answer = self.generate_prompt(prompt)

        return {"answer": answer, "video": video_link}
