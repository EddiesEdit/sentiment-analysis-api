# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .inference import load_artifacts, predict_sentiment

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0",
    description="A simple API to predict sentiment (positive/negative) using a trained NLP model."
)

# Load model and tokenizer once at startup

model, tokenizer, max_len = load_artifacts()

# Define request model
class SentimentRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API 🚀"}

@app.post("/predict")
def predict(request: SentimentRequest):
    text = request.text
    sentiment, prob = predict_sentiment(text, model, tokenizer, max_len)
    return {
        "sentiment": sentiment,
        "confidence": round(prob, 2)
    }
print("Finished loading app/main.py.")