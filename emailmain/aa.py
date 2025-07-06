import joblib

# Load saved models from disk
models = joblib.load('spam_detector_models.pkl')

# Ask for input from user
text = input("Enter the email or message to classify:\n")

# Classify using all available models
print("\nPredictions:")
for name, model in models.items():
    pred = model.predict([text])[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{name}: {label}")
