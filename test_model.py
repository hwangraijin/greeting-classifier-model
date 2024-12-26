import joblib

# Load the trained model
model = joblib.load('greeting_classifier_model.pkl')

# Test sentences
test_sentences = [
    "Good morning!",
    "How are you?",
    "Can you send the report by tomorrow?",
    "Let's meet for lunch.",
    "I have a meeting at 3 PM.",
    "Hi there!"
]

# Predict greetings
predictions = model.predict(test_sentences)

for text, prediction in zip(test_sentences, predictions):
    print(f"'{text}' => Label: {prediction}")
