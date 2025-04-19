from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests
import json
import re

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TensorFlow model
MODEL = tf.keras.models.load_model("models/1.keras")
CLASS_NAMES = [
    "HDPE (High-Density Polyethylene)",
    "OTHERS",
    "PET (polyethylene terephthalate)",
    "PP (polypropylene)",
    "PVC (Polyvinyl chloride)"
]

@app.get("/ping")
async def ping():
    return {"message": "Hello, Badi"}

# Helper to convert image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")

    return {
        "class": predicted_class,
        "confidence": confidence,
        "prediction": f"<h2>Plastic Type Prediction</h2><p><strong>Class:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {confidence:.2f}</p>"
    }

# Request model
class InsightRequest(BaseModel):
    plastic_type: str

# Clean Gemini's markdown response
def clean_json_output(response_text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    try:
        start = response_text.index('{')
        end = response_text.rindex('}') + 1
        return response_text[start:end]
    except ValueError:
        return response_text.strip()

# Insights endpoint
@app.post("/insights")
async def get_insights(request: InsightRequest):
    prompt = f"""
    Provide a JSON object with the following structure only, without extra text or markdown formatting:

    {{
      "Plastic_name": "Full name of the plastic type",
      "Common_uses": ["list of common uses"],
      "Recycling_category": "Recycling number or symbol (e.g. #1)",
      "Environmental_impact": "Concise but detailed environmental impact",
      "Alternatives": ["list of sustainable alternatives"]
    }}

    The plastic type is: {request.plastic_type}
    """

    try:
        # Gemini API request
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyDqcZHYRtufGxHuy4RGrFKe05aIUL96E6s",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        gemini_data = response.json()

        # Debug full response
        print("\nFull Gemini API response:\n", json.dumps(gemini_data, indent=2))

        if "candidates" not in gemini_data:
            print("Gemini response missing 'candidates':", gemini_data)
            return {"insight": "<p>Gemini API did not return a valid response. Please try again later.</p>"}

        try:
            raw_output = gemini_data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError) as e:
            print("Error accessing Gemini content:", e)
            return {"insight": "<p>Unexpected response format from Gemini. Please try again later.</p>"}

        print("\nGemini Raw Response:\n", raw_output)
        cleaned = clean_json_output(raw_output)
        print("\nCleaned JSON for parsing:\n", cleaned)

        structured = json.loads(cleaned)

        html_output = f"""
        <h2>{structured.get("Plastic_name", "")}</h2>
        <h3>Common Uses</h3>
        <ul>{"".join(f"<li>{use}</li>" for use in structured.get("Common_uses", []))}</ul>
        <h3>Recycling Category</h3>
        <p>{structured.get("Recycling_category", "")}</p>
        <h3>Environmental Impact</h3>
        <p>{structured.get("Environmental_impact", "")}</p>
        <h3>Alternatives</h3>
        <ul>{"".join(f"<li>{alt}</li>" for alt in structured.get("Alternatives", []))}</ul>
        """

        return {"insight": html_output}

    except json.JSONDecodeError as e:
        print("Error processing Gemini response:", e)
        return {"insight": "<p>Could not parse structured insights. Please try again later.</p>"}
    except Exception as e:
        print("General error:", e)
        return {"insight": "<p>Error generating insights. Please check your input or try again later.</p>"}if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
