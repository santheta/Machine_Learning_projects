import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Free entry in a weekly competition",
    "Hey are we meeting today?",
    "Win cash prize now",
    "Urgent call required",
    "Let's have dinner tonight"
]

labels = [1, 0, 1, 1, 0]

# FIT vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# FIT model
model = MultinomialNB()
model.fit(X, labels)

# VERIFY BEFORE SAVING (CRITICAL)
print("Before saving model:", model)

# SAVE AFTER FIT
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ New trained model saved")
