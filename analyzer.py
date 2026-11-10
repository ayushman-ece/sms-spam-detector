import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_sms_data(file_path):

    df = pd.read_csv(file_path)

    required_columns = ["label", "message"]

    # Validate required columns
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Dataset is missing required columns.")

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check missing values
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Validate labels
    valid_labels = {"ham", "spam"}

    if not set(df["label"].unique()).issubset(valid_labels):
        raise ValueError("Dataset contains invalid labels.")

    # Validate message content
    if not df["message"].apply(
        lambda message: isinstance(message, str) and bool(message.strip())
    ).all():
        raise ValueError("Dataset contains invalid messages.")

    print("SMS dataset loaded successfully!")
    print(f"Total messages: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df



def preprocess_text(message):

    # Convert text to lowercase
    message = message.lower()

    # Remove extra spaces
    message = re.sub(r"\s+", " ", message).strip()

    return message


def prepare_sms_data(df):

    df = df.copy()

    df["message"] = df["message"].apply(preprocess_text)

    print("\nText preprocessing completed!")

    return df


def convert_text_to_features(df):

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["message"])

    print("\n========== TF-IDF FEATURE EXTRACTION ==========")
    print(f"Total messages: {X.shape[0]}")
    print(f"Unique features: {X.shape[1]}")

    return X, vectorizer




def train_spam_model(X, df):

    y = df["label"]

    # Split features and labels
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create the model
    model = MultinomialNB()

    # Train the model
    model.fit(X_train, y_train)

    print("\n========== MODEL TRAINING ==========")
    print("Spam detection model trained successfully!")
    print(f"Training messages: {X_train.shape[0]}")
    print(f"Testing messages: {X_test.shape[0]}")

    return model, X_test, y_test




def evaluate_spam_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["ham", "spam"]
        )
    )

    return predictions



def predict_sms(model, vectorizer, message):

    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a valid SMS message.")

    cleaned_message = preprocess_text(message)

    message_features = vectorizer.transform([cleaned_message])

    prediction = model.predict(message_features)[0]

    print("\n========== SMS SPAM PREDICTION ==========")
    print(f"Message: {message}")
    print(f"Prediction: {prediction.upper()}")

    return prediction




def export_predictions(model, vectorizer, messages):

    cleaned_messages = [
        preprocess_text(message) for message in messages
    ]

    message_features = vectorizer.transform(cleaned_messages)

    predictions = model.predict(message_features)

    results = pd.DataFrame({
        "message": messages,
        "predicted_label": predictions
    })

    results.to_csv("spam_predictions.csv", index=False)

    print("\n========== PREDICTION REPORT ==========")
    print(results)
    print("\nPrediction report saved successfully!")

    return results


