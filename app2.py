import streamlit as st
import numpy as np
import pickle
from textblob import TextBlob
import random
import time
import requests
from streamlit_autorefresh import st_autorefresh

#autorefresh every 5 seconds for interaction history
count = st_autorefresh(interval=5000, key="auto_refresh")

CLAUSES = {
    "C1": "Payment overdue > 15 days",
    "C2": "Past missed payments",
    "C3": "Repayment risk category",
    "C4": "Customer sentiment"
}

STRATEGIES = {
    0: "Friendly Reminder",
    1: "Firm Reminder",
    2: "Assign to Telecaller",
    3: "Assign to Agent",
    4: "Legal Notification"
}

#dummy customer data
customer_data = [
    {"phoneNumber": "1234567890", "overdue_days": 22, "missed_payments": 3, "risk_category": 2, "demo": 1, "income": 20000},
    {"phoneNumber": "0987654321", "overdue_days": 5, "missed_payments": 0, "risk_category": 0, "demo": 0, "income": 50000},
    {"phoneNumber": "1112223333", "overdue_days": 17, "missed_payments": 2, "risk_category": 1, "demo": 2, "income": 30000},
    {"phoneNumber": "2223334444", "overdue_days": 10, "missed_payments": 1, "risk_category": 0, "demo": 1, "income": 45000},
    {"phoneNumber": "3334445555", "overdue_days": 20, "missed_payments": 1, "risk_category": 2, "demo": 2, "income": 35000},
    {"phoneNumber": "4445556666", "overdue_days": 12, "missed_payments": 0, "risk_category": 1, "demo": 0, "income": 60000},
    {"phoneNumber": "5556667777", "overdue_days": 8, "missed_payments": 0, "risk_category": 0, "demo": 2, "income": 75000},
    {"phoneNumber": "6667778888", "overdue_days": 5, "missed_payments": 0, "risk_category": 0, "demo": 1, "income": 50000},
    {"phoneNumber": "7778889999", "overdue_days": 25, "missed_payments": 3, "risk_category": 2, "demo": 0, "income": 20000},
    {"phoneNumber": "8889990000", "overdue_days": 7, "missed_payments": 1, "risk_category": 1, "demo": 1, "income": 48000},
    {"phoneNumber": "9990001111", "overdue_days": 15, "missed_payments": 1, "risk_category": 1, "demo": 2, "income": 55000},
    {"phoneNumber": "0001112222", "overdue_days": 2, "missed_payments": 0, "risk_category": 0, "demo": 0, "income": 32000},
    {"phoneNumber": "1112223333", "overdue_days": 28, "missed_payments": 4, "risk_category": 2, "demo": 1, "income": 18000},
    {"phoneNumber": "2223334444", "overdue_days": 14, "missed_payments": 0, "risk_category": 1, "demo": 2, "income": 46000}
]

def load_interactions_from_api():
    resp = requests.get("https://fastapi-rl.onrender.com/interactions")
    return resp.json()

class QTable:
    def __init__(self, n_actions):
        self.qtable = {}

    def update(self, data):
        self.qtable.update(data)

    def get(self, state):
        return self.qtable.get(state, np.zeros(len(STRATEGIES)))

q_table = QTable(len(STRATEGIES))

#sentiment analysis
def analyze_customer_message(message):
    sentiment = TextBlob(message).sentiment.polarity
    return round(max(-1.0, min(sentiment, 1.0)), 2)

#state generation
def generate_state():
    state = random.choice(customer_data)
    sentiment = 0  #sentiment is calculated from the message later
    return np.array([state["overdue_days"], state["missed_payments"], state["risk_category"], sentiment, state["demo"], state["income"]])

def discretize_state(state):
    overdue = int(state[0] > 15)
    missed = int(state[1])
    risk = int(state[2])
    sentiment = round(state[3] * 2) / 2
    demo = int(state[4])
    income = int(round(state[5] / 10000) * 10000)
    return (overdue, missed, risk, sentiment, demo, income)

def predict_strategy(state_vector, qtable):
    state_key = discretize_state(state_vector)
    q_values = qtable.get(state_key)
    ranked = np.argsort(q_values)[::-1]
    best_action = int(ranked[0])
    return STRATEGIES[best_action], [STRATEGIES[i] for i in ranked]

def save_q_table(qtable, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(qtable.qtable, f)

def load_q_table(filename="q_table.pkl"):
    qtable = QTable(len(STRATEGIES))
    with open(filename, "rb") as f:
        qtable.update(pickle.load(f))
    return qtable

def main():    
    st.title("Customer Interaction and Strategy Prediction")
    
    tab2, tab3 = st.tabs(["Interaction History", "Strategy Usage"])
    
    q_table_filename = "q_table_exlong.pkl"  #you can specify a custom filename if needed
    
    #load q table
    try:
        q_table = load_q_table(q_table_filename)
        st.success("Loaded existing Q-table.")
    except FileNotFoundError:
        st.warning("No Q-table found. A new Q-table will be initialized.")
    
    #interaction history tab
    with tab2:
        st.header("Interaction History")
    
        interactions = load_interactions_from_api()
        interactions = list(reversed(interactions))
    
        if not interactions:
            st.write("No interactions yet.")
        else:
            header_cols = st.columns([3, 5, 2, 2, 2])
            header_cols[0].markdown("**Phone Number**")
            header_cols[1].markdown("**Customer Response**")
            header_cols[2].markdown("**Customer Sentiment**")
            header_cols[3].markdown("**Best Strategy**")
            header_cols[4].markdown("**Ranked Strategies**")
    
            for i, item in enumerate(interactions):
                key = f"toggle_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False
    
                with st.container():
                    row_cols = st.columns([3, 5, 2, 2, 2])
                    row_cols[0].markdown(item["phoneNumber"])
                    row_cols[1].markdown(item["message"])
                    row_cols[2].markdown(f"`{item['sentiment']}`")
                    row_cols[3].markdown(item["best_strategy"])
    
                    if row_cols[4].button("View", key=f"btn_{i}"):
                        st.session_state[key] = not st.session_state[key]
    
                    #show ranked strategies
                    if st.session_state[key]:
                        st.markdown("**Ranked Strategies:**")
                        for idx, strat in enumerate(item["ranked_strategies"], start=1):
                            st.markdown(f"{idx}. {strat}")
    
                    st.markdown("---")
    
    
    #strategy usage tab
    with tab3:
        st.header("Strategy Usage")
        #interactions = st.session_state["interactions"]
        interactions = load_interactions_from_api()
    
        #get the best strategies
        best_strategies = [interaction["best_strategy"] for interaction in interactions]
    
        if len(best_strategies) == 0:
            st.write("No strategies used yet.")
        else:
            st.write(f"Displaying most used best strategies:")
            strategy_counts = {strategy: best_strategies.count(strategy) for strategy in STRATEGIES.values()}
            st.bar_chart(strategy_counts)
            st.write(strategy_counts)

if __name__ == "__main__":
    main()
