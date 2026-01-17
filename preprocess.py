import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.ensemble import IsolationForest
import joblib

stemmer = PorterStemmer()

URGENT_WORDS = [
    "urgent", "asap", "critical", "important",
    "overdue", "escalation", "priority",
    "now", "fast", "today", "quick", "quickly",
    "immediately", "right away"
]

URGENT_PHRASES = [
    "immediate action required",
    "time sensitive",
    "requires immediate attention",
    "do not delay",
    "respond immediately",
    "urgent request",
    "urgent matter",
    "action needed",
    "attention required"
]

fin_words = ["bank","account","transfer","invoice","transactions","payment"]
fin_stems = [stemmer.stem(word) for word in fin_words]

ROLE_ACTIONS = {
    "Finance": ["payment", "invoice", "transfer", "account", "bank"],
    "HR": ["onboarding", "policy", "leave", "training"],
    "Engineer": ["code", "deploy", "bug", "build"],
    "Manager": ["approval", "review", "budget", "meeting"]
}
ROLE_ACTIONS_STEMMED = {}

for role, actions in ROLE_ACTIONS.items():
    ROLE_ACTIONS_STEMMED[role] = [stemmer.stem(a) for a in actions]

def clean_text(text):
    
    text = text.lower()
    text = text.replace(".", "").replace(",", "").replace("/", "")
    return text

def compute_urgency_score(email_text):
    text = clean_text(email_text)
    words = text.split()

    if len(words) == 0:
        return 0.0

    urgency_word_count = 0
    for word in words:
        if word in URGENT_WORDS:
            urgency_word_count += 1

    urgency_phrase_count = 0
    for phrase in URGENT_PHRASES:
        if phrase in text:
            urgency_phrase_count += 1

    total_urgency_hits = urgency_word_count + urgency_phrase_count

    # Structural urgency: very short command-style emails
    if len(words) <= 8:
        total_urgency_hits += 1

    # Normalize by length
    urgency_score = total_urgency_hits / len(words)
    return urgency_score

def fin_intent(email_text):
    text = clean_text(email_text)
    words = text.split()

    if len(words) == 0:
        return 0.0

    finan_intent_count = 0
    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in fin_stems:
            finan_intent_count += 1

    return finan_intent_count / len(words)


def role_action_mismatch(email_text, sender_role):
    text = clean_text(email_text)
    words = text.split()

    if sender_role not in ROLE_ACTIONS_STEMMED:
        return 0

    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in fin_stems and sender_role != "Finance":
            return 1

    return 0

def assign_risk_level(row):
    reasons = []

    # HIGH RISK
    if row["role_mismatch"] == 1 and row["financial_intent"] > 0:
        reasons.append("Financial request from a role that normally does not handle payments")
        return "HIGH", reasons

    # MEDIUM RISK
    if row["anomaly_flag"] == 1:
        reasons.append("Email behavior deviates from normal internal patterns")

    if row["urgency_score"] > 0.1:
        reasons.append("Unusual urgency detected in the email")

    if reasons:
        return "MEDIUM", reasons

    # LOW RISK
    return "LOW", ["No suspicious behavioral patterns detected"]


def main():
    data = pd.read_csv("corporate_mails.csv")
    data["urgency_score"] = data["email_body"].apply(compute_urgency_score)
    data["financial_intent"] = data["email_body"].apply(fin_intent)
    data["role_mismatch"] = data.apply(lambda row: role_action_mismatch(row["email_body"], row["sender_role"]),axis=1)

    #train model
    cols = ["urgency_score","financial_intent","role_mismatch"]
    x = data[cols]
    model = IsolationForest(n_estimators=100,random_state=42,contamination=0.2)
    model.fit(x)
    data["anomaly_pred"] = model.predict(x)
    data["anomaly_flag"] = data["anomaly_pred"].apply(lambda x : 1 if x==-1 else 0)
    data["anomaly_score"] = model.decision_function(x)

    print(pd.crosstab(data["anomaly_flag"], data["is_bec"]))

    risk_results = data.apply(assign_risk_level, axis=1)
    data["risk_level"] = risk_results.apply(lambda x: x[0])
    data["risk_reasons"] = risk_results.apply(lambda x: x[1])
    
    #save the model
    model_bundle = {"model":model,"stemmer":stemmer,"fin_stems" : fin_stems,"feature_cols":cols}
    joblib.dump(model_bundle,"bec_model.pkl")
    print("model saved!!")

if __name__ == "__main__":
    main()
