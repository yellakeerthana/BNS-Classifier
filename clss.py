import pandas as pd
import numpy as np
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class BNSClassifier:
    """
    Pure Dataset-Driven BNS Section Classifier
    Uses BM25 ranking only.
    No rule-based logic.
    No AI.
    """

    # Common legal boilerplate words to ignore
    LEGAL_NOISE = {
        "section", "act", "person", "whoever",
        "offence", "offences", "punishment",
        "law", "shall", "within", "thereof",
        "under", "against", "means", "includes"
    }

    # -------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------
    def __init__(self, csv_path: str = "bns_sections.csv"):
        print("Loading BNS dataset...")

        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()

        # Normalize possible inconsistent column names
        self.df.rename(columns={
            "Section _name": "Section Name",
            "Chapter_subtype": "Chapter Subtype",
            "Chapter_name": "Chapter Name"
        }, inplace=True)

        # Fill missing values safely
        for col in ["Section Name", "Chapter Subtype", "Description"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")

        # Ensure Section column numeric
        self.df["Section"] = pd.to_numeric(self.df["Section"], errors="coerce")
        self.df = self.df.dropna(subset=["Section"])

        # Combine dataset text ONLY (no manual logic)
        self.df["combined_text"] = (
            self.df.get("Section Name", "").astype(str) + " " +
            self.df.get("Chapter Subtype", "").astype(str) + " " +
            self.df.get("Description", "").astype(str)
        )

        # Tokenize entire dataset
        self.corpus_tokens = [
            self._tokenize(text)
            for text in self.df["combined_text"].tolist()
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus_tokens)

        print("BNS Classifier ready (Pure BM25 Mode).\n")

    # -------------------------------------------------
    # TOKENIZER (Improved Cleaning)
    # -------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()

        # Remove stopwords + legal noise + very short words
        filtered = [
            word for word in tokens
            if word not in ENGLISH_STOP_WORDS
            and word not in self.LEGAL_NOISE
            and len(word) > 2
        ]

        return filtered

    # -------------------------------------------------
    # PURE BM25 CLASSIFICATION
    # -------------------------------------------------
    def classify_complaint(self, complaint: str, top_k: int = 5) -> Dict:

        if not complaint or not complaint.strip():
            return {"error": "Empty complaint provided"}

        query_tokens = self._tokenize(complaint)

        # Compute BM25 similarity scores
        scores = self.bm25.get_scores(query_tokens)

        # Rank sections
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "section_number": int(self.df.iloc[idx]["Section"]),
                "section_name": self.df.iloc[idx].get("Section Name", ""),
                "chapter_subtype": self.df.iloc[idx].get("Chapter Subtype", ""),
                "description": self.df.iloc[idx].get("Description", ""),
                "similarity_score": round(float(scores[idx]), 4)
            })

        # Confidence normalization (0–1)
        max_score = max(scores) if len(scores) > 0 else 0
        confidence = round(max_score / (max_score + 1), 4) if max_score > 0 else 0.0

        return {
            "bns_sections": results,
            "confidence_score": confidence,
            "dataset_used": True,
            "rule_based": False,
            "ai_model_used": False
        }


# -------------------------------------------------
# CLI TESTING MODE
# -------------------------------------------------
if __name__ == "__main__":

    classifier = BNSClassifier()

    while True:
        complaint = input("Enter complaint (or type 'exit'): ")

        if complaint.lower() == "exit":
            print("Exiting...")
            break

        result = classifier.classify_complaint(complaint)

        print("\n🔎 BNS Section Suggestions\n")
        print("Confidence Score:", result["confidence_score"])
        print("-" * 60)

        for i, sec in enumerate(result["bns_sections"], 1):
            print(f"{i}. Section {sec['section_number']}")
            print(f"   Name        : {sec['section_name']}")
            print(f"   Chapter     : {sec['chapter_subtype']}")
            print(f"   Score       : {sec['similarity_score']}")
            print("-" * 60)

        print("\n")