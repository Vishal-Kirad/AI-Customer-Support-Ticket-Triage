# 🎫 AI Customer Support Ticket Triage

An end-to-end Artificial Intelligence project that automatically analyzes customer support tickets and predicts the **support queue** and **ticket priority** using Natural Language Processing (NLP) and machine learning.

The project includes a **Streamlit dashboard** for real-time predictions and a **human-in-the-loop review mechanism** for low-margin predictions.

---

## 🚀 Project Demo

The application allows a user to enter:

- Ticket Subject
- Ticket Body
- Ticket Type

and predicts:

- **Support Queue**
- **Priority**
- **SVM Decision Margin**
- **Human Review Required / Automatic Prediction**

### Example

A production outage ticket was correctly routed to:

- **Support Queue:** Service Outages and Maintenance
- **Priority:** High

The application also accepted the prediction because both decision margins were above the configured review threshold.

---

## 🎯 Problem Statement

Customer support teams receive a large number of tickets every day. Manually reading, categorizing, prioritizing, and routing these tickets can be time-consuming.

This project builds an AI-based ticket triage system that reads incoming support tickets and automatically predicts:

1. Which support queue should handle the ticket.
2. How urgent the ticket is.

Low-margin predictions are flagged for human review instead of being automatically accepted.

---

## 🧠 Solution Overview

The system follows this pipeline:

```text
Customer Support Ticket
          │
          ├── Subject
          ├── Body
          └── Type
          │
          ▼
     Text Cleaning
          │
          ▼
   Subject + Body
          │
          ▼
       TF-IDF
          │
          ├───────────────┐
          │               │
          ▼               ▼
       Queue           Priority
       Model             Model
          │               │
          └───────┬───────┘
                  ▼
          Prediction + Decision Margin
                  │
          ▼
       Human Review Check
```

The Queue and Priority predictions are made using **separate LinearSVC classifiers**.

---

## 📊 Dataset

The project uses the:

**Customer IT Support - Ticket Dataset**

Dataset source:

https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

Important dataset fields:

| Column | Description |
|---|---|
| `subject` | Customer ticket subject |
| `body` | Customer ticket message |
| `type` | Ticket type |
| `queue` | Support queue / target |
| `priority` | Priority / target |
| `language` | Ticket language |

The project uses English-language tickets for the final modeling workflow.

---

## 🧹 Text Preprocessing

The ticket subject and body are combined into a single text field.

The preprocessing pipeline includes:

- Lowercasing
- Removing newline and tab characters
- Removing URLs
- Removing email addresses
- Removing punctuation
- Removing extra whitespace

Technical terms are retained rather than applying aggressive stop-word removal.

---

## 🔢 Feature Engineering

### TF-IDF

The final text representation uses TF-IDF with:

```text
max_features = 30000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

### Ticket Type

The categorical `type` feature is converted using:

```text
OneHotEncoder(
    handle_unknown="ignore"
)
```

The TF-IDF and one-hot encoded features are combined using sparse matrix operations.

---

## 🤖 Machine Learning Model

The project uses **Linear Support Vector Machines (LinearSVC)** for multiclass classification.

Two separate classifiers are trained:

```text
Queue
TF-IDF + Ticket Type (OHE)
        ↓
LinearSVC
        ↓
Support Queue
```

```text
Priority
TF-IDF + Ticket Type (OHE)
        ↓
LinearSVC
        ↓
Priority
```

The train/test data is created using a common joint stratification based on the Queue + Priority combination.

Example:

```python
stratify_label = (
    df["queue"].astype(str) + "_" +
    df["priority"].astype(str)
)
```

This ensures that both prediction tasks are evaluated using the same held-out ticket population.

---

## 📈 Model Performance

The final reported test-set performance for the selected model is:

### Queue Classification

| Metric | Score |
|---|---:|
| Accuracy | **65.67%** |
| Macro F1 | **66.42%** |
| Weighted F1 | **65.68%** |

### Priority Classification

| Metric | Score |
|---|---:|
| Accuracy | **67.78%** |
| Macro F1 | **66.81%** |
| Weighted F1 | **67.70%** |

Evaluation includes:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Class-wise performance

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

### Dashboard Inputs

- Ticket Subject
- Ticket Body
- Ticket Type

### Dashboard Outputs

- Predicted Support Queue
- Predicted Priority
- Decision Margin for Queue
- Decision Margin for Priority
- Human Review status

Run the dashboard with:

```bash
streamlit run app.py
```

---

## 👤 Human-in-the-Loop Review

The system includes a human-review mechanism for uncertain predictions.

Because `LinearSVC` does not directly produce calibrated probabilities, the dashboard uses the model's `decision_function()` output.

For multiclass predictions, the current decision-margin signal is based on the difference between the highest and second-highest decision scores.

### Important

The decision margin is **not a percentage probability**.

For example:

```text
Decision Margin = 0.617
```

does **not** mean 61.7% probability.

A lower margin means that the model is less decisive between competing classes and can therefore be routed for human review.

Low-margin tickets are logged in:

```text
human_review_log.csv
```

---

## 📋 Example Predictions

### Production Outage

**Subject:**
```text
Production server is completely down
```

**Type:**
```text
Incident
```

**Prediction:**
```text
Queue: Service Outages and Maintenance
Priority: High
```

This example was successfully routed to the outage queue with a high-priority prediction.

---

### Duplicate Payment

**Subject:**
```text
I was charged twice for the same subscription
```

**Type:**
```text
Incident
```

**Prediction:**
```text
Queue: Billing and Payments
Priority: Medium
```

The dashboard accepted this prediction based on the configured decision-margin threshold.

---

## 📁 Project Structure

```text
AI-Customer-Support-Ticket-Triage/
│
├── app.py
├── FinalSTP.ipynb
├── newSTP.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── tfidf.pkl
├── ohe.pkl
├── queue_svm.pkl
└── priority_svm.pkl
```

### Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `FinalSTP.ipynb` | Final model development notebook |
| `newSTP.ipynb` | Additional experimentation |
| `tfidf.pkl` | Trained TF-IDF vectorizer |
| `ohe.pkl` | Trained One-Hot Encoder |
| `queue_svm.pkl` | Trained Queue classifier |
| `priority_svm.pkl` | Trained Priority classifier |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignored files |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd AI-Customer-Support-Ticket-Triage
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- SciPy
- TF-IDF
- One-Hot Encoding
- LinearSVC
- Streamlit
- Jupyter Notebook
- Pickle

---

## 🔮 Future Improvements

Possible improvements include:

- Calibrated probability estimates
- Transformer / sentence-embedding models
- Confusion matrix visualization
- REST API deployment
- Database-backed ticket storage
- Real-time ticket ingestion
- Human-review feedback loop
- Automatic model retraining from reviewed tickets
- Cloud deployment

---

## 📌 Project Highlights

- End-to-end NLP classification pipeline
- Multiclass Queue prediction
- Multiclass Priority prediction
- Common joint-stratified evaluation
- TF-IDF based text representation
- One-Hot Encoding for categorical ticket type
- Separate ML models for Queue and Priority
- Interactive Streamlit dashboard
- Human-in-the-loop review mechanism
- Low-margin ticket logging
- Reproducible project structure

---

## 👨‍💻 Author

**Ajay**

Artificial Intelligence Capstone Project

---

## 📄 Project Submission

This repository contains the source code, trained model artifacts, notebooks, and application required to reproduce and run the project.

For the project report/demo, refer to the GitHub repository and the submitted project PDF.
