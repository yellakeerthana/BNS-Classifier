from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

# Import your custom modules
from ocr import extract_text_from_image
from extractor import extract_entities
from bns_classifier import suggest_bns_v2

app = Flask(__name__)

# Setup Uploads
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""
    extracted_data = {}
    bns_sections = []

    if request.method == "POST":
        # 1. Get Inputs
        complaint_text = request.form.get("complaint_text")
        file = request.files.get("complaint_image")

        # 2. Handle Image OCR or Direct Text
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            extracted_text = extract_text_from_image(filepath)
        elif complaint_text:
            extracted_text = complaint_text

        # 3. Processing Pipeline
        if extracted_text:
            # STEP A: Run BNS Classifier first to get the Legal Category
            bns_sections = suggest_bns_v2(extracted_text)
            
            # STEP B: Determine the Crime Type from the top BNS result
            crime_label = "General Criminal Matter"
            if bns_sections and len(bns_sections) > 0:
                top_match=bns_sections[0]
                # We pull the 'chapter' field returned by your Gemini prompt
                crime_label = top_match.get('chapter', "General Criminal Matter")

            # STEP C: Pass the Chapter Subtype into the Extractor
            extracted_data = extract_entities(extracted_text, crime_category=crime_label)

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        extracted_data=extracted_data,
        bns_sections=bns_sections
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)