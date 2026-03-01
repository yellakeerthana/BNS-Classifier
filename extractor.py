import spacy
import os
import re

# Load the Spacy model
try:
    nlp = spacy.load("en_core_web_md") 
except:
    # Auto-download if missing
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text, crime_category=None):
    """
    Extracts legal entities and includes the BNS Chapter Subtype 
    passed from the classifier.
    """
    doc = nlp(text)

    # 1. AI Entity Extraction
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC", "ORG"]]
    times = [ent.text for ent in doc.ents if ent.label_ in ["TIME", "DATE"]]
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


    time_patterns = [
        r"\byesterday\b",
        r"\blast night\b",
        r"\blast week\b",
        r"\blast month\b",
        r"\b\d{1,2} (am|pm|AM|PM)\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d+ days ago\b",
        r"\bearly morning\b",
        r"\baround \d{1,2}\b"
    ]
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        times.extend(matches)

    # 2. Keyword Backup for Locations (helpful for Indian addresses)
    
    
    if not locations:
        place_indicators = ["near", "at", "in","on"]
        words=text.split()
        for i, word in enumerate(words):
            if word.lower() in place_indicators:
                phrase = []
                for j in range(i, min(i + 5, len(words))):
                    phrase.append(words[j])
                locations.append(" ".join(phrase))
                break
    # 3. Data Formatting for the UI Boxes
    data = {
        "Crime_Type": crime_category if crime_category else "General Legal Matter",
        "Location": ", ".join(list(set(locations))) if locations else "Not Mentioned",
        "Time": ", ".join(list(set(times))) if times else "Not Mentioned",
        "Persons": ", ".join(list(set(persons))) if persons else "Not Mentioned",
        "Summary": text[:150] + "..." if len(text) > 150 else text
    }

    return data