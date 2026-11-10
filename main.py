
from analyzer import (
    load_sms_data,
    prepare_sms_data,
    convert_text_to_features,
    train_spam_model,
    evaluate_spam_model,
    predict_sms,
    export_predictions
)


def main():

    try:
        print("======================================")
        print("       SMS SPAM DETECTION SYSTEM")
        print("======================================")

        # Step 1: Load dataset
        df = load_sms_data("spam_data.csv")

        # Step 2: Preprocess messages
        df = prepare_sms_data(df)

        # Step 3: Convert text into numerical features
        X, vectorizer = convert_text_to_features(df)

        # Step 4: Train the model
        model, X_test, y_test = train_spam_model(X, df)

        # Step 5: Evaluate the model
        evaluate_spam_model(model, X_test, y_test)

        # Step 6: Predict new messages
        predict_sms(
            model,
            vectorizer,
            "Congratulations! You won a free prize. Claim now!"
        )

        predict_sms(
            model,
            vectorizer,
            "Hey, are you coming to college tomorrow?"
        )

        # Step 7: Export prediction results
        messages = [
            "Congratulations! You won a free mobile phone!",
            "Are you coming to college tomorrow?",
            "Claim your cash reward immediately!",
            "Please send me the assignment."
        ]

        export_predictions(model, vectorizer, messages)

        print("\n======================================")
        print("    PROJECT COMPLETED SUCCESSFULLY")
        print("======================================")

    except FileNotFoundError:
        print("Error: spam_data.csv was not found.")

    except (ValueError, KeyError) as error:
        print(f"Data validation error: {error}")

    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()
