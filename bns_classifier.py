import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
#import google.generativeai as genai
from google import genai
import os

# ==========================================
# 1. SETUP - PASTE YOUR API KEY HERE
# ==========================================
GENAI_API_KEY = "AIzaSyCWwfjYAddNGtjaOYEZSLWrtdO23CQ7MhA" 
client = genai.Client(api_key=GENAI_API_KEY)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
# Load the dataset
df = pd.read_csv('bns_sections.csv')
df.columns = df.columns.str.strip()

# Fill missing values to prevent errors during encoding
for col in ["Section _name", "Chapter_subtype", "Description"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Ensure Section is numeric and drop invalid rows
df["Section"] = pd.to_numeric(df["Section"], errors="coerce")
df = df.dropna(subset=["Section"])

# Create the search context
df['contextual_info'] = (
    df['Section _name'].astype(str) + "  " +
    df['Chapter_subtype'].astype(str) + "  "+
    df['Description'].astype(str)
)

# Initialize the Semantic Search Model
# Note: HF_TOKEN warnings can be safely ignored
model = SentenceTransformer('all-MiniLM-L6-v2')
encoded_sections = model.encode(df['contextual_info'].tolist(), convert_to_tensor=True)

# ==========================================
# 3. THE BNS CLASSIFIER FUNCTION
# ==========================================
def suggest_bns_v2(user_complaint):
    try:
        # STEP A: Local Semantic Search (Sentence Transformer)
        # This finds the top 5 most similar sections in your CSV mathematically
        query_embed = model.encode(user_complaint, convert_to_tensor=True)
        scores = util.cos_sim(query_embed, encoded_sections)[0]
        
        # We take 5 candidates so the AI has enough context to choose the right one
        top_results = torch.topk(scores, k=10)
        
        candidates = []
        for score, idx in zip(top_results[0], top_results[1]):
            row = df.iloc[idx.item()]
            candidates.append({
                "section": str(int(row['Section'])),
                "name": row['Section _name'],
                "chapter": row['Chapter_subtype'],
                "description": row['Description'][:300], # Provide context for AI
                "raw_score": round(score.item() * 100, 2)
            })

        # STEP B: AI Reasoning (Gemini)
        prompt = f"""
        You are a Bharatiya Nyaya Sanhita (BNS) Legal Expert. 
        Analyze the complaint and the candidate sections provided.
        
        COMPLAINT: "{user_complaint}"
        
        CANDIDATES FROM DATABASE:
        {json.dumps(candidates)}
        
        TASK:
        1. Select the MOST relevant BNS sections. Ignore candidates that don't fit the facts.
        2. Extract details: Persons, Location, Time, and a Summary.
        3. Provide a 'Reason' for your selection.
        
        IMPORTANT: Return the output as valid JSON ONLY. Do not add any conversational text.
        
        OUTPUT FORMAT:
        Return ONLY valid JSON in this format:
        IMPORTANT:
        The "chapter" field MUST contain the Chapter_subtype from the database.
        [
            {{
                "section": "number",
                "section_name": "name",
                "chapter": "copy the Chapter_subtype from candidates",
                "confidence": "use the number from candidates"
            }}
        ]
        """

        response = client.models.generate_content(model="gemini-1.5-flash",content=prompt)
        
        # Clean the response (removes ```json blocks if Gemini adds them)
        raw_text = response.text.strip().replace('```json', '').replace('```', '')

        return json.loads(raw_text)

    except Exception as e:
        # Return a fallback if the API fails
        return candidates[:5]
        

