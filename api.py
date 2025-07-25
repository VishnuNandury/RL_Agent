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

    #templates = {
    #    "Friendly Reminder": "Hi, this is a gentle payment reminder.",
    #    "Firm Reminder": "Please be advised that your payment is overdue.",
    #    "Assign to Telecaller": "Our team will be in touch shortly.",
    #    "Assign to Agent": "A recovery agent will contact you soon.",
    #    "Legal Notification": "This is a legal warning for overdue payment."
    #}
    #templates = {
    #    "Friendly Reminder": "Write a warm and respectful message reminding the customer of their overdue payment and offering help if needed. Keep it under 3 lines or 250 characters.",
    #    "Firm Reminder": "Write a professional but firm message informing the customer that their payment is overdue and should be paid soon. Limit to 3 lines or 250 characters.",
    #    "Assign to Telecaller": "Inform the customer in a courteous tone that a representative will contact them shortly regarding their overdue payment. Keep the message under 2 lines or 200 characters.",
    #    "Assign to Agent": "Compose a serious, formal message notifying the customer that a field recovery agent has been assigned due to continued non-payment. Keep it under 3 lines or 250 characters.",
    #    "Legal Notification": "Write an assertive, formal message warning the customer about possible legal action due to overdue payments. Limit it to 3 lines or 250 characters."
    #}
    templates = {
        "Friendly Reminder": (
            "You are a customer support assistant for a financial services provider.\n"
            "Write a polite and respectful message to a known customer reminding them of their overdue payment.\n"
            "- Keep the tone warm but professional, not overly casual.\n"
            "- Mention that payment is overdue and offer help if needed.\n"
            "- Limit to 2–3 sentences and use newlines for readability."
        ),
        "Firm Reminder": (
            "You are a professional support agent communicating with an existing customer.\n"
            "Write a firm but courteous message stating that their payment is overdue and must be settled soon.\n"
            "- Avoid overly friendly language; keep the tone serious and businesslike.\n"
            "- Limit to 2–3 sentences and format with newlines."
        ),
        "Assign to Telecaller": (
            "You are informing a customer that a representative will follow up.\n"
            "Craft a short message stating that a team member will reach out to discuss their overdue payment.\n"
            "- Keep the tone respectful and brief.\n"
            "- Use 1–2 sentences.\n"
            "- Format with newlines if needed."
        ),
        "Assign to Agent": (
            "You are notifying a customer that a field recovery agent has been assigned.\n"
            "Write a serious and formal message that:\n"
            "- Clearly states the reason (continued non-payment)\n"
            "- Communicates the next step (agent visit)\n"
            "- Uses 2–3 sentences.\n"
            "- Maintain a firm and official tone."
        ),
        "Legal Notification": (
            "You are a compliance officer sending a legal warning to a known customer.\n"
            "Write a formal, assertive message warning of legal action due to overdue payments.\n"
            "- Avoid emotional or friendly language.\n"
            "- Use 2–3 concise sentences.\n"
            "- Format for clarity and seriousness."
        )
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
