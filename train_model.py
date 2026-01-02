import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]

# Rename columns for clarity
data.columns = ["label", "text"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["text"]
y = data["label"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorizer
# -----------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# -----------------------------
# Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Save trained objects
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and Vectorizer saved successfully")
