# agents/rag_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class MedicalRAG:
    def __init__(self, csv_path: str = "rag/DiseaseAndSymptoms.csv"):
        self.csv_path = csv_path
        self.vectorizer = TfidfVectorizer()
        self.disease_symptoms_map = []
        self.tfidf_matrix = None
        self.diseases = []
        self.symptoms_text = []
        
        self._load_and_index()

    def _load_and_index(self):
        """Loads the CSV and creates a searchable symptom index."""
        if not os.path.exists(self.csv_path):
            print(f"⚠️ [RAG Warning] Could not find {self.csv_path}. RAG will return empty context.")
            return

        df = pd.read_csv(self.csv_path)
        
        # Combine all symptom columns into a single string per disease
        symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
        df['combined_symptoms'] = df[symptom_cols].fillna('').agg(' '.join, axis=1)
        df['combined_symptoms'] = df['combined_symptoms'].str.replace('_', ' ').str.replace('  ', ' ')

        # Store for retrieval
        self.diseases = df['Disease'].values
        self.symptoms_text = df['combined_symptoms'].values
        
        # Build TF-IDF Matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.symptoms_text)
        print(f"📚 [RAG Engine] Indexed {len(self.diseases)} disease profiles.")

    def retrieve(self, extracted_symptoms: list[str], top_k: int = 3) -> str:
        """Finds the most likely diseases based on NER symptoms."""
        if self.tfidf_matrix is None or not extracted_symptoms:
            return "No specific medical context found."

        query = " ".join(extracted_symptoms)
        query_vec = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        context_lines = ["Possible matching conditions based on knowledge base:"]
        seen_diseases = set()
        
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.05:  # Similarity threshold
                disease = self.diseases[idx].strip()
                if disease not in seen_diseases:
                    symptoms = self.symptoms_text[idx].strip()
                    context_lines.append(f"- {disease} (Typical symptoms: {symptoms})")
                    seen_diseases.add(disease)
        
        if len(seen_diseases) == 0:
            return "No strong matches found in the knowledge base."
            
        return "\n".join(context_lines)