import spacy
import os

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

    # 2. Keyword Backup for Locations (helpful for Indian addresses)
    keywords = text.lower().split()
    place_indicators = ["near", "at", "road", "market", "street", "station", "area"]
    if not locations:
        for i, word in enumerate(keywords):
            if word in place_indicators and i+1 < len(keywords):
                locations.append(keywords[i] + " " + keywords[i+1])

    # 3. Data Formatting for the UI Boxes
    data = {
        "Crime_Type": crime_category if crime_category else "General Legal Matter",
        "Location": ", ".join(list(set(locations))) if locations else "Not Mentioned",
        "Time": ", ".join(list(set(times))) if times else "Not Mentioned",
        "Persons": ", ".join(list(set(persons))) if persons else "Not Mentioned",
        "Summary": text[:150] + "..." if len(text) > 150 else text
    }

    return data