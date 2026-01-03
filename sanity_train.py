import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Starting training...")

texts = [
    "Free entry in a weekly competition",
    "Hey are we meeting today?",
    "Win cash prize now",
    "Urgent call required",
    "Let's have dinner tonight"
]

labels = [1, 0, 1, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("Vectorizer fitted")

model = MultinomialNB()
print("Model BEFORE fit:", model)

model.fit(X, labels)
print("Model AFTER fit:", model)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Saved model.pkl and vectorizer.pkl")
