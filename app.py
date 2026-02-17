from flask import Flask, render_template, request
from services.gemini_service import analyze_drug_interaction

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    medicines = request.form.get("medicines")
    result = analyze_drug_interaction(medicines)

    return render_template(
        "result.html",
        medicines=medicines,
        severity=result["severity"],
        risk=result["risk_explanation"],
        recommendation=result["medical_recommendation"]
    )

if __name__ == "__main__":
    app.run(debug=True)
