import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Has class_count_:", hasattr(model, "class_count_"))
print("Has feature_log_prob_:", hasattr(model, "feature_log_prob_"))
print("Vectorizer vocab size:", len(vectorizer.vocabulary_))
