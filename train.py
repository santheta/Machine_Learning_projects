import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data (replace with your real dataset)
texts = [
    "Free entry in a weekly competition",
    "Hey are we meeting today?",
    "Win cash prize now",
    "Call now to claim reward",
    "Let's have lunch tomorrow"
]

labels = [1, 0, 1, 1, 0]   # 1 = spam, 0 = ham

# 1️⃣ FIT VECTORRIZER
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 2️⃣ FIT MODEL
model = MultinomialNB()
model.fit(X, labels)

# 3️⃣ SAVE AFTER FITTING (VERY IMPORTANT)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Fresh trained model saved")
