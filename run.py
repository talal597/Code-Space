# Email Phishing Detection System
# Description: Detect phishing emails using ML and NLP with interpretability tools.

#IMORTANT NOTE: TO RUN THIS CODE USE GOOGLE COLLAB FIRST RUN THIS CODE THEN RUN THESE LINES IN SEQUENCE:
#!pip install imbalanced-learn xgboost shap beautifulsoup4
#run_training(input_file='spam.csv', show_plots=True)
#run_prediction("Congratulations! You've won $1000! Click here to claim your prize!")
#run_prediction("Hi, this is a normal email about our meeting tomorrow.")

import pandas as pd
import numpy as np
import os
import re
import argparse
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           precision_recall_curve, f1_score, precision_score, recall_score)
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ===================== Data Loading =====================
def load_data(path):
    """Load and standardize email dataset"""
    try:
        df = pd.read_csv(path, encoding='latin-1')
        # Handle different dataset formats
        if 'v1' in df.columns and 'v2' in df.columns:
            # spam.csv format
            df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'body'})
            df['subject'] = ''
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        elif 'label' in df.columns and 'text' in df.columns:

            df = df.rename(columns={'text': 'body'})
            if 'subject' not in df.columns:
                df['subject'] = ''

        # Handle missing values
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')

        df = df.drop_duplicates(subset=['body'])

        print(f"Loaded dataset with {len(df)} samples")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
# ===================== Keyword Baseline Classifier =====================
class KeywordBaseline(BaseEstimator, ClassifierMixin):
    """Simple keyword-based phishing detector for baseline comparison"""
    def __init__(self):
        self.phishing_keywords = [
            'urgent', 'verify', 'confirm', 'suspended', 'update', 'click here',
            'act now', 'limited time', 'expire', 'winner', 'congratulations',
            'free', 'cash', 'money', 'bank', 'credit card', 'password',
            'login', 'account', 'security', 'alert', 'warning', 'immediate'
        ]
        self.classes_ = np.array([0, 1])
    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for text in X:
            text_lower = str(text).lower()
            keyword_count = sum(1 for keyword in self.phishing_keywords if keyword in text_lower)
            # If 2 or more suspicious keywords found, classify as phishing
            predictions.append(1 if keyword_count >= 2 else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        predictions = self.predict(X)
        # Simple probability assignment
        proba = np.zeros((len(predictions), 2))
        for i, pred in enumerate(predictions):
            if pred == 1:
                proba[i] = [0.3, 0.7]  # 70% confidence for phishing
            else:
                proba[i] = [0.8, 0.2]  # 80% confidence for ham
        return proba
# ===================== Text Preprocessing =====================
def clean_text(text):
    """Advanced text cleaning with NLP preprocessing"""
    if pd.isna(text) or text == '':
        return ''
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        text = ' '.join(words)
    except:
        pass  # If NLTK fails, continue without stopword removal
    return text

def extract_urls(text):
    """Extract URLs from text"""
    if pd.isna(text):
        return []
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, str(text))

def has_suspicious_domain(urls):
    """Check for suspicious domain patterns"""
    suspicious_patterns = [
        'bit.ly', 'tinyurl', 'shortened', 'click', 'free',
        'paypal', 'amazon', 'google', 'microsoft', 'apple'  # Common phishing targets
    ]
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
            for pattern in suspicious_patterns:
                if pattern in domain:
                    return True
        except:
            continue
    return False
# ===================== Feature Engineering =====================
def extract_advanced_features(df):
    """Extract comprehensive features for phishing detection"""
    df['raw_html'] = df['body']
    df['num_html_tags'] = df['raw_html'].apply(lambda x: len(BeautifulSoup(str(x), 'html.parser').find_all()))
    df['has_html_tags'] = (df['num_html_tags'] > 0).astype(int)
    # URL-based features
    df['urls'] = df['raw_html'].apply(extract_urls)
    df['num_urls'] = df['urls'].apply(len)
    df['has_urls'] = (df['num_urls'] > 0).astype(int)
    df['has_suspicious_domain'] = df['urls'].apply(has_suspicious_domain).astype(int)
    # Clean text for lexical analysis
    df['body_clean'] = df['body'].apply(clean_text)
    df['subject_clean'] = df['subject'].apply(clean_text)
    df['combined_text'] = df['subject_clean'] + ' ' + df['body_clean']
    # Lexical features
    df['word_count'] = df['body_clean'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['body_clean'].apply(lambda x: len(str(x)))
    df['punctuation_count'] = df['raw_html'].apply(lambda x: len(re.findall(r'[!?.,;:]', str(x))))

    # Capitalization patterns
    df['caps_ratio'] = df['raw_html'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    df['excessive_caps'] = (df['caps_ratio'] > 0.3).astype(int)

    # Exclamation marks (urgency indicator)
    df['exclamation_count'] = df['raw_html'].apply(lambda x: str(x).count('!'))
    df['excessive_exclamation'] = (df['exclamation_count'] > 3).astype(int)

    print("Feature engineering completed!")
    return df
def vectorize_text(df, max_features=5000):
    """Create TF-IDF vectors from cleaned text"""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95
    )
    X_tfidf = tfidf.fit_transform(df['combined_text'])
    # Save vectorizer
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(tfidf, 'outputs/tfidf_vectorizer.joblib')
    print(f"TF-IDF vectorization completed with {X_tfidf.shape[1]} features")
    return X_tfidf, tfidf
# ===================== Model Training & Evaluation =====================
def perform_cross_validation(model, X, y, cv_folds=5):
    """Perform stratified k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = {
        'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
        'precision': cross_val_score(model, X, y, cv=skf, scoring='precision'),
        'recall': cross_val_score(model, X, y, cv=skf, scoring='recall'),
        'f1': cross_val_score(model, X, y, cv=skf, scoring='f1')
    }
    print("Cross-Validation Results:")
    for metric, scores in cv_scores.items():
        print(f"  {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return cv_scores

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Phishing'],
                yticklabels=['Ham', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'outputs/Confusion_Matrix_{model_name.replace(" ", "_")}.png')
    plt.close()
    return cm

def evaluate_model(model, X_test, y_test, name, show_plots=False):
    """Comprehensive model evaluation"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)
    # Basic metrics
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS - {name}")
    print(f"{'='*50}")
    print(f"Classification Report:\n{classification_report(y_test, y_preds)}")
    # Confusion Matrix
    cm = plot_confusion_matrix(y_test, y_preds, name)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Analysis:")
    print(f"  True Negatives (Ham correctly identified): {tn}")
    print(f"  False Positives (Ham misclassified as Phishing): {fp}")
    print(f"  False Negatives (Phishing missed): {fn}")
    print(f"  True Positives (Phishing correctly identified): {tp}")

    # Additional metrics
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    print(f"\nKey Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 4))
    # ROC subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision-Recall subplot
    plt.subplot(1, 2, 2)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall_curve, precision_curve, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/Evaluation_Curves_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def hyperparameter_tuning(model_type, X_train, y_train):
    """Perform hyperparameter tuning for models"""
    print(f"Performing hyperparameter tuning for {model_type}...")
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:  # Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best recall score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def extract_feature_importance(model, feature_names, model_name):
    """Extract and plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'outputs/Feature_Importance_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        feature_imp_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        feature_imp_df.to_csv(f'outputs/Feature_Importance_{model_name.replace(" ", "_")}.csv', index=False)

        print(f"Top 10 important features for {model_name}:")
        for i in range(min(10, len(indices))):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
# ===================== SHAP Explainability =====================
def explain_with_shap(model, X, feature_names, model_name, sample_size=100):
    """Generate SHAP explanations"""
    try:
        print(f"Generating SHAP explanations for {model_name}...")
        sample_indices = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
        X_sample = X[sample_indices]
        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        # Summary plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'outputs/SHAP_Summary_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP explanations saved for {model_name}")
    except Exception as e:
        print(f"SHAP explanation failed for {model_name}: {e}")

# ===================== Main Pipeline =====================
def main_pipeline(args):
    """Main training pipeline"""
    print("Starting Email Phishing Detection Training Pipeline")
    print("="*60)
    # Load and preprocess data
    df = load_data(args.input)
    df = extract_advanced_features(df)
    # Create feature matrix
    X_text, tfidf = vectorize_text(df)
    # Lexical and engineered features
    feature_columns = [
        'word_count', 'char_count', 'punctuation_count', 'num_html_tags',
        'has_html_tags', 'num_urls', 'has_urls', 'has_suspicious_domain',
        'caps_ratio', 'excessive_caps', 'exclamation_count', 'excessive_exclamation'
    ]
    X_features = df[feature_columns].values
    X_combined = hstack([X_text, X_features]).tocsr()
    # Create feature names for interpretability
    tfidf_features = [f"tfidf_{i}" for i in range(X_text.shape[1])]
    all_feature_names = tfidf_features + feature_columns
    y = df['label']

    # Initial train-test split
    X_train_initial, X_test, y_train_initial, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    # Apply SMOTE to training data only
    print("\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_initial, y_train_initial)
    print(f"Training set after SMOTE: {y_train_balanced.value_counts().to_dict()}")
    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=1)
    }
    # Add keyword baseline
    models["Keyword Baseline"] = KeywordBaseline()
    results = {}
    # Train and evaluate models
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"TRAINING AND EVALUATING: {name}")
        print(f"{'='*60}")

        # Train model
        if name == "Keyword Baseline":
            # Baseline uses original text
            baseline_text = df['combined_text'].values
            X_train_text = baseline_text[y_train_initial.index] if hasattr(y_train_initial, 'index') else baseline_text[:len(y_train_initial)]
            X_test_text = baseline_text[y_test.index] if hasattr(y_test, 'index') else baseline_text[len(y_train_initial):]

            model.fit(X_train_text, y_train_initial)

            # Evaluate on test set
            results[name] = evaluate_model(model, X_test_text, y_test, name, show_plots=args.show_plots)
        else:
            # Hyperparameter tuning for advanced models
            if args.tune_hyperparams and name != "Logistic Regression":
                model = hyperparameter_tuning(name.replace(" ", ""), X_train_balanced, y_train_balanced)
            else:
                model.fit(X_train_balanced, y_train_balanced)
            # Cross-validation on balanced training set
            perform_cross_validation(model, X_train_balanced, y_train_balanced)
            # Save model
            joblib.dump(model, f"outputs/{name.replace(' ', '_')}.joblib")
            # Evaluate on test set
            results[name] = evaluate_model(model, X_test, y_test, name, show_plots=args.show_plots)
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                extract_feature_importance(model, all_feature_names, name)
            # SHAP explanations
            if args.shap_analysis:
                explain_with_shap(model, X_test, all_feature_names, name)

    # Generate comparison report
    generate_comparison_report(results)
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE COMPLETED!")
    print("Check the 'outputs' folder for all results, plots, and saved models.")
    print(f"{'='*60}")

def generate_comparison_report(results):
    """Generate comprehensive model comparison report"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")

    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    print(comparison_df.to_string(index=False))
    # Save comparison
    comparison_df.to_csv('outputs/Model_Comparison_Report.csv', index=False)
    # Find best model for each metric
    print(f"\nBest Models by Metric:")
    print(f"  Best Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']} ({comparison_df['Precision'].max():.4f})")
    print(f"  Best Recall: {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']} ({comparison_df['Recall'].max():.4f})")
    print(f"  Best F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']} ({comparison_df['F1-Score'].max():.4f})")
    print(f"  Best ROC-AUC: {comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']} ({comparison_df['ROC-AUC'].max():.4f})")

def predict_message(model_path, message):
    """Predict if a message is phishing or ham"""
    try:
        # Load model and vectorizer
        model = joblib.load(model_path)
        tfidf = joblib.load('outputs/tfidf_vectorizer.joblib')
        # Preprocess message
        clean_msg = clean_text(message)
        # Extract features
        urls = extract_urls(message)
        features = {
            'word_count': len(clean_msg.split()),
            'char_count': len(clean_msg),
            'punctuation_count': len(re.findall(r'[!?.,;:]', message)),
            'num_html_tags': len(BeautifulSoup(message, 'html.parser').find_all()),
            'has_html_tags': int(len(BeautifulSoup(message, 'html.parser').find_all()) > 0),
            'num_urls': len(urls),
            'has_urls': int(len(urls) > 0),
            'has_suspicious_domain': int(has_suspicious_domain(urls)),
            'caps_ratio': sum(1 for c in message if c.isupper()) / max(len(message), 1),
            'excessive_caps': int(sum(1 for c in message if c.isupper()) / max(len(message), 1) > 0.3),
            'exclamation_count': message.count('!'),
            'excessive_exclamation': int(message.count('!') > 3)
        }
        # Create feature vector
        X_tfidf = tfidf.transform([clean_msg])
        X_features = np.array([[features[col] for col in [
            'word_count', 'char_count', 'punctuation_count', 'num_html_tags',
            'has_html_tags', 'num_urls', 'has_urls', 'has_suspicious_domain',
            'caps_ratio', 'excessive_caps', 'exclamation_count', 'excessive_exclamation'
        ]]])
        X_combined = hstack([X_tfidf, X_features]).tocsr()

        # Make prediction
        prediction = model.predict(X_combined)[0]
        probability = model.predict_proba(X_combined)[0]
        label = "PHISHING" if prediction == 1 else "HAM"
        confidence = probability[prediction]

        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Message: {message[:100]}{'...' if len(message) > 100 else ''}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probability [Ham, Phishing]: [{probability[0]:.3f}, {probability[1]:.3f}]")
        # Show key features
        print(f"\nKey Features Detected:")
        for key, value in features.items():
            if value > 0:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure you have trained the model first!")

# ===================== Entry Point =====================
def run_training(input_file='spam.csv', show_plots=False, tune_hyperparams=False, shap_analysis=False):
    """Run the training pipeline - Colab-friendly version"""
    class Args:
        def __init__(self):
            self.input = input_file
            self.show_plots = show_plots
            self.tune_hyperparams = tune_hyperparams
            self.shap_analysis = shap_analysis
    args = Args()
    main_pipeline(args)

def run_prediction(message, model_path='outputs/Random_Forest.joblib'):
    """Run prediction - Colab-friendly version"""
    predict_message(model_path=model_path, message=message)

if __name__ == '__main__':
    import sys
    # Check if running in Colab/Jupyter (has kernel arguments)
    if any('kernel' in arg for arg in sys.argv):
        print("Detected Colab/Jupyter environment!")
        print("Use the functions directly:")
        print("  run_training() - to train models")
        print("  run_prediction('your message here') - to make predictions")
        print("\nExample:")
        print("  run_training(input_file='spam.csv', show_plots=True)")
    else:
        # Regular command line usage
        parser = argparse.ArgumentParser(description="Email Phishing Detection System")
        parser.add_argument('--input', type=str, default='spam.csv',
                           help='Input CSV file path (default: spam.csv)')
        parser.add_argument('--show-plots', action='store_true',
                           help='Display plots interactively')
        parser.add_argument('--predict', type=str,
                           help='Enter a message to classify as PHISHING or HAM')
        parser.add_argument('--model', type=str, default='outputs/Random_Forest.joblib',
                           help='Model path for prediction (default: Random Forest)')
        parser.add_argument('--tune-hyperparams', action='store_true',
                           help='Perform hyperparameter tuning (slower but better results)')
        parser.add_argument('--shap-analysis', action='store_true',
                           help='Generate SHAP explanations for model interpretability')
        args = parser.parse_args()

        if args.predict:
            predict_message(model_path=args.model, message=args.predict)
        else:
            main_pipeline(args)
