import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import google.generativeai as genai

genai.configure(api_key="AIzaSyBif5c4kQOeJKpo-aRNQva86h1ldss_ggE")  
model = genai.GenerativeModel("gemini-2.0-flash")

def get_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

urgent_keywords = ["immediately", "urgent", "critical", "asap", "cannot access", "blocked"]

def get_priority(text):
    if not isinstance(text, str):
        return "Not Urgent"
    text = text.lower()
    for word in urgent_keywords:
        if word in text:
            return "Urgent"
    return "Not Urgent"

def extract_contact_info(text):
    if not isinstance(text, str):
        return ""
    phones = re.findall(r'\+?\d[\d -]{8,}\d', text)
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return {"phones": phones, "emails": emails}

def generate_reply(email_subject, email_body, sentiment, priority):
    prompt = f"""
    You are a customer support assistant.
    Email Subject: {email_subject}
    Email Body: {email_body}
    Sentiment: {sentiment}
    Priority: {priority}

    Task: Write a professional, empathetic, and context-aware reply to the customer.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error generating reply: {e}]"

st.set_page_config(page_title="AI-Powered Email Assistant", layout="wide")
st.title("📩 AI-Powered Communication Assistant")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["sentiment"] = df["body"].apply(get_sentiment)
    df["priority"] = df.apply(
        lambda row: "Urgent" if (get_priority(row["subject"]) == "Urgent" or get_priority(row["body"]) == "Urgent") else "Not Urgent",
        axis=1
    )
    df["contact_info"] = df["body"].apply(extract_contact_info)
    df["ai_reply"] = df.apply(
        lambda row: generate_reply(row["subject"], row["body"], row["sentiment"], row["priority"]),
        axis=1
    )

    st.sidebar.header("Filters")
    sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", options=df["sentiment"].unique(), default=df["sentiment"].unique())
    priority_filter = st.sidebar.multiselect("Filter by Priority", options=df["priority"].unique(), default=df["priority"].unique())

    filtered_df = df[(df["sentiment"].isin(sentiment_filter)) & (df["priority"].isin(priority_filter))]

    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Emails", len(df))
        st.metric("Urgent Emails", (df["priority"] == "Urgent").sum())
    with col2:
        st.metric("Positive", (df["sentiment"] == "Positive").sum())
        st.metric("Negative", (df["sentiment"] == "Negative").sum())

    st.subheader("📈 Visual Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df["priority"].value_counts())
    with col2:
        st.bar_chart(df["sentiment"].value_counts())

    st.subheader("📬 Processed Emails")
    for i, row in filtered_df.iterrows():
        with st.expander(f"{row['subject']} | {row['priority']} | {row['sentiment']}"):
            st.write(f"**From:** {row['sender']}")
            st.write(f"**Date:** {row['sent_date']}")
            st.write(f"**Body:** {row['body']}")
            st.write(f"**Contact Info:** {row['contact_info']}")
            reply = st.text_area("AI Suggested Reply", row["ai_reply"], key=f"reply_{i}")
            if st.button(f"✅ Approve & Send Reply #{i}"):
                st.success(f"Reply approved for email {i}")
