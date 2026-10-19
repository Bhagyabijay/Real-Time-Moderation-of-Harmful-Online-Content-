# app.py
import streamlit as st
import numpy as np
import joblib

# --- Load Saved Model and Vectorizer ---
model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
thresholds = joblib.load("thresholds.pkl")  # Thresholds you saved earlier

# --- Prediction Function ---
def classify_message(message):
    X_tfidf = vectorizer.transform([message])
    probs = model.predict_proba(X_tfidf)[0]

    if probs[0] >= thresholds[0]:  # Hate Speech threshold
        return "Hate Speech", probs
    elif probs[1] >= thresholds[1]:  # Offensive threshold
        return "Offensive", probs
    else:
        return "Neither", probs

# --- Streamlit UI ---
st.title("Real-time Chat Moderation System")
st.write("Only safe messages will be sent. Offensive or hateful messages will be blocked with alerts.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat input
new_message = st.text_input("Enter your message:")

if st.button("Send Message") and new_message.strip():
    label, probs = classify_message(new_message)
    
    if label == "Neither":
        # Allow sending
        st.session_state["chat_history"].append((new_message, label))
    else:
        # Block message and show toast alert
        st.toast(
            f"⚠️ Your message was blocked: {label}", 
            icon="🚨"
        )


# Display chat history (only safe messages)
st.subheader("Chat History")
for msg, label in st.session_state["chat_history"]:
    st.markdown(f"<span style='color:green;'>[{label}]</span> {msg}", unsafe_allow_html=True)
