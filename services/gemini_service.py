from google import genai
import json


API_KEY = "AIzaSyANAGIpNEikQ9n3uT2iRD5mBX1I2K1TqQY"

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3-flash-preview"

def analyze_drug_interaction(medicines):

    prompt = f"""
    A patient is taking the following medicines:
    {medicines}

    Return output ONLY in valid JSON format:

    {{
        "severity": "HIGH | MODERATE | LOW",
        "risk_explanation": "Clear explanation of interaction",
        "medical_recommendation": "Professional medical advice"
    }}
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return json.loads(response.text)

    except Exception:
        return {
            "severity": "UNKNOWN",
            "risk_explanation": "Unable to analyze interactions due to system issue.",
            "medical_recommendation": "Please consult a healthcare professional immediately."
        }
