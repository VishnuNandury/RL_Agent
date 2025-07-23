from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from textblob import TextBlob
import pickle
from utils import save_interaction, load_interactions
from app2 import STRATEGIES, customer_data, QTable, discretize_state, predict_strategy

app = FastAPI()

class WebhookRequest(BaseModel):
    messageType: str
    message: str
    phoneNumber: str

def analyze_customer_message(message):
    sentiment = TextBlob(message).sentiment.polarity
    return round(max(-1.0, min(sentiment, 1.0)), 2)

def load_q_table(filename="q_table_exlong.pkl"):
    qtable = QTable(len(STRATEGIES))
    with open(filename, "rb") as f:
        qtable.update(pickle.load(f))
    return qtable

q_table = load_q_table()

@app.get("/interactions")
async def get_interactions():
    return load_interactions()

@app.post("/rlmessage")
async def webhook(req: WebhookRequest):
    if req.messageType.lower() == "init":
        return {"response": "Your payment is due. Please pay at the earliest convenience."}

    customer = next((c for c in customer_data if c["phoneNumber"] == req.phoneNumber), None)
    if not customer:
        return {"error": "Customer not found"}

    sentiment = analyze_customer_message(req.message)
    state_vec = np.array([
        customer["overdue_days"], customer["missed_payments"], customer["risk_category"],
        sentiment, customer["demo"], customer["income"]
    ])

    best, ranked = predict_strategy(state_vec, q_table)

    templates = {
        "Friendly Reminder": "Hi, this is a gentle payment reminder.",
        "Firm Reminder": "Please be advised that your payment is overdue.",
        "Assign to Telecaller": "Our team will be in touch shortly.",
        "Assign to Agent": "A recovery agent will contact you soon.",
        "Legal Notification": "This is a legal warning for overdue payment."
    }

    save_interaction({
        "phoneNumber": req.phoneNumber,
        "message": req.message,
        "sentiment": sentiment,
        "ranked_strategies": ranked,
        "best_strategy": best
    })

    return {
        "best_strategy": best,
        "template_message": templates.get(best, "")
    }
