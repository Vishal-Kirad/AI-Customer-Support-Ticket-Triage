
import streamlit as st
import pandas as pd
import re
import pickle
from scipy.sparse import hstack
import os
from datetime import datetime



st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="centered"
)



@st.cache_resource
def load_models():

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("ohe.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("queue_svm.pkl", "rb") as f:
        queue_model = pickle.load(f)

    with open("priority_svm.pkl", "rb") as f:
        priority_model = pickle.load(f)

    return tfidf, encoder, queue_model, priority_model


tfidf, encoder, queue_model, priority_model = load_models()



def clean_text(text):

    text = str(text).lower()

    # Remove newline and tab characters
    text = re.sub(r'[\n\r\t]+', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# SVM Confidence

def get_svm_confidence(model, features):

    scores = model.decision_function(features)

    if scores.ndim == 1:

        confidence = abs(scores[0])

    else:

        sorted_scores = sorted(scores[0], reverse=True)

        confidence = sorted_scores[0] - sorted_scores[1]

    return confidence


# Human Review Logging

def log_for_human_review(
    subject,
    body,
    ticket_type,
    predicted_queue,
    predicted_priority,
    queue_confidence,
    priority_confidence
):

    log_file = "human_review_log.csv"

    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "body": body,
        "type": ticket_type,
        "predicted_queue": predicted_queue,
        "predicted_priority": predicted_priority,
        "queue_confidence": round(queue_confidence, 4),
        "priority_confidence": round(priority_confidence, 4)
    }])

    if os.path.exists(log_file):

        row.to_csv(
            log_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        row.to_csv(
            log_file,
            index=False
        )


# Dashboard

st.title("🎫 AI Customer Support Ticket Triage")

st.write(
    "Enter a customer support ticket to predict its "
    "support queue and priority."
)

st.divider()



subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Unable to login to my account"
)

body = st.text_area(
    "Ticket Body",
    placeholder="Enter the customer's support request here...",
    height=200
)

ticket_type = st.selectbox(
    "Ticket Type",
    [
        "Incident",
        "Request",
        "Problem",
        "Change"
    ]
)


# Prediction

if st.button("Predict Ticket", type="primary"):

    if not subject.strip() or not body.strip():

        st.warning(
            "Please enter both subject and ticket body."
        )

    else:


        clean_subject = clean_text(subject)
        clean_body = clean_text(body)

        text = clean_subject + " " + clean_body


        # TF-IDF

        text_tfidf = tfidf.transform([text])


        # One-Hot Encode Ticket Type

        type_encoded = encoder.transform(
            pd.DataFrame({
                "type": [ticket_type]
            })
        )


        final_features = hstack([
            text_tfidf,
            type_encoded
        ])

        predicted_queue = queue_model.predict(
            final_features
        )[0]

        predicted_priority = priority_model.predict(
            final_features
        )[0]


        # Confidence Scores


        queue_confidence = get_svm_confidence(
            queue_model,
            final_features
        )

        priority_confidence = get_svm_confidence(
            priority_model,
            final_features
        )

        # Human Review Decision


        CONFIDENCE_THRESHOLD = 0.5

        human_review = (
            queue_confidence < CONFIDENCE_THRESHOLD
            or
            priority_confidence < CONFIDENCE_THRESHOLD
        )


        # Results
 

        st.divider()

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Support Queue**")
            st.info(predicted_queue)

        with col2:

            st.write("**Priority**")
            st.success(predicted_priority.upper())


        # Confidence


        st.write(
            f"Queue confidence score: "
            f"`{queue_confidence:.3f}`"
        )

        st.write(
            f"Priority confidence score: "
            f"`{priority_confidence:.3f}`"
        )

        # Human Review
  

        if human_review:

            st.warning(
                "⚠️ Low-confidence prediction — "
                "Human Review Required"
            )

            # Log ONLY low-confidence predictions
            log_for_human_review(
                subject,
                body,
                ticket_type,
                predicted_queue,
                predicted_priority,
                queue_confidence,
                priority_confidence
            )

            st.info(
                "This ticket has been added to "
                "`human_review_log.csv`."
            )

        else:

            st.success(
                "✅ Prediction confidence is acceptable."
            )
