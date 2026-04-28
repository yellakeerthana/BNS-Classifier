from google import genai
from google.genai import types

API_KEY = "AIzaSyAHuAF73h7Vb7ovf3P8rgMk4qSiMLn0kws"

client = genai.Client(api_key=API_KEY)


def extract_text_from_image(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = """
    Read the complaint written in this image.
    Extract the text exactly as written.
    Only return the complaint text.
    """

    response = client.models.generate_content(
        model="models/gemini-3-flash-preview",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )

    return response.text