 
# This notebook demonstrates building a predictive model to classify emails as spam or ham (not spam) using various machine learning algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    classification_report, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 1. Data Loading and Exploration

# %%
# Load the dataset
url = "https://raw.githubusercontent.com/commitit/Spam-Email-Detection/main/spam.csv"
data = pd.read_csv(url, encoding='latin-1')

# Drop unnecessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Display basic info
print(f"Dataset shape: {data.shape}")
data.head()

# %%
# Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=data)
plt.title('Distribution of Spam vs Ham Emails')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.show()

# Class percentages
print(data['label'].value_counts(normalize=True))

# %% [markdown]
# ## 2. Data Preprocessing

# %%
# Convert labels to binary (0 for ham, 1 for spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords and stem
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_text)

# Show example
print("Original text:", data['text'][0])
print("Processed text:", data['processed_text'][0])

# %% [markdown]
# ## 3. Feature Extraction

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], data['label'], 
    test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Train shape: {X_train_tfidf.shape}")
print(f"Test shape: {X_test_tfidf.shape}")

# %% [markdown]
# ## 4. Model Training and Evaluation

# %%
# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='linear', probability=True)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Train
    model.fit(X_train_tfidf, y_train)
    
    # Predict
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_tfidf, y_train, cv=5).mean()
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_score': cv_score,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Print metrics
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"5-Fold CV Accuracy: {cv_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# %% [markdown]
# ## 5. Model Comparison

# %%
# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': results.keys(),
    'Test Accuracy': [results[name]['accuracy'] for name in results],
    'CV Accuracy': [results[name]['cv_score'] for name in results]
}).sort_values('Test Accuracy', ascending=False)

# Display comparison
comparison

# %%
# Plot model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Test Accuracy', y='Model', data=comparison, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0.9, 1.0)
plt.show()

# %% [markdown]
# ## 6. Detailed Evaluation of Best Model

# %%
# Get best model
best_model_name = comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']
y_pred = results[best_model_name]['y_pred']
y_prob = results[best_model_name]['y_prob']

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 7. Creating a Prediction Pipeline

# %%
# Create a pipeline with preprocessing and best model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', best_model)
])

# Fit on entire training data
pipeline.fit(X_train, y_train)

# Test the pipeline
sample_emails = [
    "Congratulations! You've won a $1000 gift card. Click here to claim!",  # Spam
    "Hi John, just checking in about our meeting tomorrow at 2pm.",  # Ham
    "URGENT: Your account has been compromised. Verify your details now!",  # Spam
    "Thanks for your email. I'll get back to you by end of day."  # Ham
]

predictions = pipeline.predict(sample_emails)
probabilities = pipeline.predict_proba(sample_emails)

for email, pred, prob in zip(sample_emails, predictions, probabilities):
    print(f"\nEmail: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}")
    print(f"Probability (Ham: {prob[0]:.4f}, Spam: {prob[1]:.4f})")

# %% [markdown]
# ## 8. Saving the Model

# %%
import joblib

# Save the pipeline
joblib.dump(pipeline, 'spam_classifier.pkl')

# To load later:
# loaded_pipeline = joblib.load('spam_classifier.pkl')
