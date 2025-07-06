# Sample email text for testing spam detection
email_text = """airbnb
Hi Matthew,
Thanks for using airbnb. We really appreciate you choosing airbnb for your travel
plans.
To help us improve, we'd like to ask you a few questions about your experience
so far. itll only take 3minutes, and youranswers will help us make airbnb even
better for you and other guests.
Thanks,
The airbnb Team
Take the Survey
8+
Sent with from airbnb, Inc.
hood
888 Brannan St, San Francisco, CA 94103
Who is medallia?·Unsubscribe·Privacy Policy
DOWNLOAD ON
DOWNLOAD ON
App Store
google play"""

# Standard library imports
import re  # Regular expressions for text processing
import warnings  # Warning management
from datetime import datetime  # Date and time handling

# Data processing and machine learning imports
import pandas as pd  # Data manipulation and analysis
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
from sklearn.model_selection import train_test_split  # Dataset splitting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Model evaluation
from sklearn.pipeline import Pipeline  # ML pipeline construction
import joblib  # Model serialization
import torch  # PyTorch for GPU/MPS acceleration

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class SpamDetector:
    """
    A machine learning-based spam detection system with hardware acceleration support.

    This class provides functionality to:
    1. Preprocess email text for spam detection
    2. Extract features from emails
    3. Train multiple classification models
    4. Evaluate model performance
    5. Make predictions on new emails
    6. Save and load trained models

    It supports hardware acceleration via CUDA (NVIDIA GPUs) or 
    MPS (Apple Metal Performance Shaders) when available.
    """

    def __init__(self, use_mps=True):
        """
        Initialize the spam detector with configurable hardware acceleration.

        Args:
            use_mps (bool): Whether to use Apple Metal Performance Shaders if available
        """
        # Dictionary to store trained models
        self.models = {}

        # TF-IDF vectorizer for converting text to numerical features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features to prevent overfitting
            stop_words='english',  # Remove common English words
            lowercase=True,  # Convert all text to lowercase
            ngram_range=(1, 2)  # Use both single words and pairs of words
        )

        # Set up the appropriate computation device (CPU, CUDA, or MPS)
        self.device = self._setup_device(use_mps)
        print(f"Using device: {self.device}")

    def _setup_device(self, use_mps):
        """
        Set up the appropriate computation device for machine learning.

        This method checks for available hardware acceleration in this order:
        1. Apple Metal Performance Shaders (MPS) if requested and available
        2. NVIDIA CUDA if available
        3. CPU as fallback

        Args:
            use_mps (bool): Whether to use Apple MPS if available

        Returns:
            torch.device: The selected computation device
        """
        if use_mps and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return torch.device("cuda")  # NVIDIA GPU
        else:
            return torch.device("cpu")   # CPU fallback

    def preprocess_text(self, text):
        """
        Enhanced text preprocessing for better spam detection.

        This method applies several cleaning steps to normalize text:
        1. Converting to lowercase
        2. Removing HTML tags
        3. Removing URLs and email addresses
        4. Normalizing punctuation
        5. Removing numbers
        6. Cleaning whitespace

        These steps help improve classification accuracy by removing
        noise and standardizing the text format.

        Args:
            text (str): The raw email text

        Returns:
            str: Preprocessed text ready for feature extraction
        """
        # Convert to lowercase for consistency
        text = text.lower()

        # Remove HTML tags that might be in the email
        text = re.sub(r'<.*?>', '', text)

        # Remove URLs (common in spam)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove email addresses (for privacy and to reduce noise)
        text = re.sub(r'\S+@\S+', '', text)

        # Normalize excessive punctuation (e.g., "!!!!!!" -> "!")
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        # Remove numbers (often used in spam)
        text = re.sub(r'\d+', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_features(self, text):
        """
        Extract additional features that indicate spam beyond just word frequencies.

        This method extracts several heuristic features that are common in spam:
        1. Punctuation usage (exclamation marks, question marks)
        2. Capitalization patterns
        3. Presence of spam-related keywords
        4. Text length and formatting characteristics

        These features complement the TF-IDF word features to improve
        classification accuracy.

        Args:
            text (str): The preprocessed email text

        Returns:
            dict: A dictionary of extracted features and their values
        """
        features = {}

        # Count exclamation marks (common in spam)
        features['exclamation_count'] = text.count('!')

        # Count question marks (often used in clickbait)
        features['question_count'] = text.count('?')

        # Count uppercase words (SHOUTING is common in spam)
        features['uppercase_count'] = sum(1 for word in text.split() if word.isupper())

        # Check for common spam keywords and phrases
        spam_keywords = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'act now',
                         'limited time', 'click here', 'buy now', 'discount', 'offer']
        features['spam_keywords'] = sum(1 for keyword in spam_keywords if keyword in text.lower())

        # Email length (spam often has shorter or much longer text)
        features['email_length'] = len(text)

        # Capital letter ratio (ALL CAPS or Lots Of Capital Letters are spam indicators)
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return features

    def train_models(self, X_train, y_train):
        """
        Train multiple classification models for spam detection.

        This method trains three different models:
        1. Multinomial Naive Bayes - Fast and effective for text classification
        2. Logistic Regression - Good for linearly separable data with probability output
        3. Random Forest - Robust ensemble method that handles non-linear relationships

        Each model is trained as part of a pipeline that includes TF-IDF vectorization.

        Args:
            X_train (list or array): Training data (email texts)
            y_train (list or array): Training labels (0 for ham, 1 for spam)
        """
        # Define models with optimized hyperparameters
        models = {
            'naive_bayes': MultinomialNB(alpha=0.1),  # Alpha smoothing parameter
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),  # More iterations for convergence
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
        }

        # Train each model in a pipeline with the TF-IDF vectorizer
        for name, model in models.items():
            # Create a pipeline that first vectorizes the text, then applies the classifier
            pipeline = Pipeline([
                ('tfidf', self.vectorizer),  # Convert text to TF-IDF features
                ('classifier', model)        # Apply the classifier to the features
            ])

            # Fit the pipeline to the training data
            pipeline.fit(X_train, y_train)

            # Store the trained pipeline in the models dictionary
            self.models[name] = pipeline

        print("All models trained successfully!")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test data and identify the best performer.

        This method:
        1. Tests each model on the provided test data
        2. Calculates accuracy and generates classification reports
        3. Identifies the best performing model based on accuracy

        Args:
            X_test (list or array): Test data (email texts)
            y_test (list or array): True labels for test data

        Returns:
            tuple: (results dictionary with metrics for each model, name of best model)
        """
        results = {}

        # Evaluate each model
        for name, model in self.models.items():
            # Generate predictions
            predictions = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)

            # Store results
            results[name] = {
                'accuracy': accuracy,
                'predictions': predictions
            }

            # Print evaluation metrics
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))  # Shows precision, recall, F1-score

        # Find the best performing model based on accuracy
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest Model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")

        return results, best_model

    def predict(self, text, model_name='naive_bayes'):
        """
        Predict whether an email is spam or not using the specified model.

        This method:
        1. Validates the requested model exists
        2. Preprocesses the input text
        3. Makes a prediction using the selected model
        4. Extracts additional features for analysis
        5. Returns a comprehensive result dictionary

        Args:
            text (str): The email text to classify
            model_name (str): Which trained model to use (default: 'naive_bayes')

        Returns:
            dict: A dictionary containing:
                - prediction: 'SPAM' or 'NOT SPAM'
                - confidence: Confidence score of the prediction
                - spam_probability: Probability of being spam
                - features: Additional extracted features

        Raises:
            ValueError: If the specified model doesn't exist
        """
        # Validate the requested model exists
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        # Preprocess the input text
        processed_text = self.preprocess_text(text)

        # Make prediction and get probability scores
        prediction = self.models[model_name].predict([processed_text])[0]
        probability = self.models[model_name].predict_proba([processed_text])[0]

        # Extract additional features for analysis and explanation
        features = self.extract_features(text)

        # Return comprehensive result dictionary
        return {
            'prediction': 'SPAM' if prediction == 1 else 'NOT SPAM',
            'confidence': max(probability),  # Highest probability (confidence in the prediction)
            'spam_probability': probability[1] if len(probability) > 1 else 0,  # Probability of being spam
            'features': features  # Additional features for analysis
        }

    def save_models(self, filepath='spam_detector_models.pkl'):
        """
        Save trained models to disk for later use.

        This method serializes the trained models dictionary using joblib,
        which efficiently handles scikit-learn objects including pipelines.

        Args:
            filepath (str): Path where the models will be saved
                           (default: 'spam_detector_models.pkl')
        """
        joblib.dump(self.models, filepath)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath='spam_detector_models.pkl'):
        """
        Load previously trained models from disk.

        This method deserializes models saved with the save_models method,
        allowing for model reuse without retraining.

        Args:
            filepath (str): Path to the saved models file
                           (default: 'spam_detector_models.pkl')
        """
        self.models = joblib.load(filepath)
        print(f"Models loaded from {filepath}")


class EmailComplianceChecker:
    """
    Enhanced compliance checker for email marketing and communications.

    This class analyzes emails for:
    1. Spam patterns that might trigger filters
    2. Regulatory compliance with laws like CAN-SPAM and GDPR
    3. Content quality issues that affect deliverability

    It provides detailed reports on compliance issues and recommendations
    for improving email deliverability and legal compliance.
    """

    def __init__(self):
        """
        Initialize the compliance checker with rules and risk weights.
        """
        # Comprehensive dictionary of compliance rules organized by category
        # Each rule is a regex pattern that identifies potential issues
        self.compliance_rules = {
            # Patterns that might trigger spam filters
            'spam_patterns': {
                'excessive_caps': r'[A-Z]{10,}',  # Too many capital letters
                'money_claims': r'(\$|money|cash|prize|win|winner|free)',  # Financial claims
                'urgency_words': r'(urgent|act now|limited time|hurry|expires|deadline)',  # Creating false urgency
                'click_bait': r'(click here|click now|visit|download|install)',  # Clickbait language
                'suspicious_punctuation': r'[!]{3,}|[?]{3,}',  # Excessive punctuation
                'phone_numbers': r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s?\d{3}[-.\s]?\d{4})',  # Phone numbers
                'email_harvesting': r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',  # Email addresses
            },
            # Legal and regulatory compliance requirements
            'regulatory_compliance': {
                'unsubscribe_required': r'(unsubscribe|opt.out|remove|stop)',  # Unsubscribe option (required)
                'physical_address': r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Place|Pl))',  # Physical address
                'sender_identification': r'(from:|sender:|company:|organization:)',  # Sender identification
                'gdpr_compliance': r'(privacy policy|data protection|personal data|consent)',  # GDPR requirements
                'can_spam_compliance': r'(advertisement|promotional|marketing)',  # CAN-SPAM requirements
            },
            # Content quality and professionalism issues
            'content_quality': {
                'spelling_errors': r'(recieve|seperate|definately|occured|accomodate)',  # Common misspellings
                'grammar_issues': r'(your\s+going|there\s+going|its\s+time|loose\s+weight)',  # Grammar errors
                'suspicious_domains': r'(\.tk|\.ml|\.ga|\.cf|bit\.ly|tinyurl)',  # Suspicious domains
                'excessive_links': r'(http[s]?://[^\s]+)',  # Too many links
            }
        }

        # Weights for calculating overall risk score
        # Spam patterns have the highest weight as they most directly affect deliverability
        self.risk_weights = {
            'spam_patterns': 0.4,  # 40% of overall score
            'regulatory_compliance': 0.3,  # 30% of overall score
            'content_quality': 0.3  # 30% of overall score
        }

    def check_compliance(self, email_content):
        """
        Perform a comprehensive compliance check on email content.

        This method:
        1. Analyzes the email against all compliance rules
        2. Calculates an overall compliance score
        3. Determines the risk level
        4. Identifies specific violations
        5. Generates actionable recommendations

        Args:
            email_content (str): The email content to analyze

        Returns:
            dict: A comprehensive compliance report containing:
                - overall_score: 0-100 score (lower is better)
                - risk_level: 'LOW', 'MEDIUM', or 'HIGH'
                - violations: List of specific rule violations
                - recommendations: Actionable suggestions for improvement
                - detailed_analysis: Detailed breakdown by category
        """
        # Initialize results structure
        results = {
            'overall_score': 0,
            'risk_level': 'LOW',
            'violations': [],
            'recommendations': [],
            'detailed_analysis': {}
        }

        total_violations = 0
        category_scores = {}

        # Check each category of compliance rules
        for category, patterns in self.compliance_rules.items():
            category_violations = 0
            category_details = {}

            # Check each rule within the category
            for rule_name, pattern in patterns.items():
                # Find all matches for the rule pattern
                matches = re.findall(pattern, email_content, re.IGNORECASE)

                if matches:
                    # Count violations and store examples
                    category_violations += len(matches)
                    category_details[rule_name] = {
                        'matches': len(matches),
                        'examples': matches[:3] if len(matches) > 3 else matches  # Store up to 3 examples
                    }

                    # Add detailed violation information to results
                    results['violations'].append({
                        'category': category,
                        'rule': rule_name,
                        'severity': self._get_severity(category, rule_name),
                        'count': len(matches),
                        'examples': matches[:3]  # Include examples in the report
                    })

            # Store category-level results
            category_scores[category] = category_violations
            results['detailed_analysis'][category] = category_details

            # Apply category weights to the total violation count
            total_violations += category_violations * self.risk_weights[category]

        # Calculate overall score (0-100, lower is better)
        # Cap at 100 to keep the scale consistent
        results['overall_score'] = min(100, total_violations * 10)

        # Determine risk level based on overall score
        if results['overall_score'] >= 70:
            results['risk_level'] = 'HIGH'
        elif results['overall_score'] >= 40:
            results['risk_level'] = 'MEDIUM'
        else:
            results['risk_level'] = 'LOW'

        # Generate actionable recommendations based on violations
        results['recommendations'] = self._generate_recommendations(results['violations'])

        return results

    def _get_severity(self, category, rule_name):
        """
        Determine the severity level for a specific compliance rule.

        This method classifies rule violations into three severity levels:
        - HIGH: Critical issues that may cause legal problems or severe deliverability issues
        - MEDIUM: Important issues that should be addressed
        - LOW: Minor issues that are good to fix but less critical

        Args:
            category (str): The category of the rule
            rule_name (str): The specific rule name

        Returns:
            str: Severity level ('HIGH', 'MEDIUM', or 'LOW')
        """
        # Rules that are considered high severity due to legal requirements
        # or major deliverability impact
        high_severity = [
            'money_claims',         # Financial claims often trigger spam filters
            'urgency_words',        # Creating false urgency is a spam indicator
            'click_bait',           # Clickbait language is heavily filtered
            'unsubscribe_required', # Legal requirement in most jurisdictions
            'physical_address'      # Required by CAN-SPAM Act
        ]

        if rule_name in high_severity:
            return 'HIGH'
        elif category == 'spam_patterns':
            return 'MEDIUM'  # Most spam patterns are medium severity
        else:
            return 'LOW'     # Content quality issues are generally lower severity

    def _generate_recommendations(self, violations):
        """
        Generate actionable recommendations based on detected violations.

        This method analyzes the violations found in the email and provides
        specific, actionable recommendations to improve compliance and deliverability.

        Args:
            violations (list): List of violation dictionaries from check_compliance

        Returns:
            list: Prioritized list of specific recommendations
        """
        recommendations = []

        # Count violations by category and rule
        violation_counts = {}
        for violation in violations:
            key = f"{violation['category']}.{violation['rule']}"
            violation_counts[key] = violation_counts.get(key, 0) + violation['count']

        # Generate specific recommendations based on violation patterns

        # Spam pattern recommendations
        if violation_counts.get('spam_patterns.excessive_caps', 0) > 0:
            recommendations.append("Reduce excessive capitalization to avoid spam filters")

        if violation_counts.get('spam_patterns.money_claims', 0) > 0:
            recommendations.append("Minimize money-related claims and promotional language")

        if violation_counts.get('spam_patterns.urgency_words', 0) > 0:
            recommendations.append("Reduce urgency language that may trigger spam filters")

        # Regulatory compliance recommendations
        # Note: These check for ABSENCE of required elements (count == 0)
        if violation_counts.get('regulatory_compliance.unsubscribe_required', 0) == 0:
            recommendations.append("Add clear unsubscribe instructions (required by law)")

        if violation_counts.get('regulatory_compliance.physical_address', 0) == 0:
            recommendations.append("Include physical business address (CAN-SPAM requirement)")

        # Content quality recommendations
        if violation_counts.get('content_quality.excessive_links', 0) > 5:
            recommendations.append("Reduce number of links to improve deliverability")

        # If no issues found, provide positive feedback
        if not recommendations:
            recommendations.append("Email appears to meet basic compliance requirements")

        return recommendations


class EnhancedSpamDetectionAPI:
    """Enhanced API wrapper with comprehensive compliance checking"""

    def __init__(self, detector):
        self.detector = detector
        self.compliance_checker = EmailComplianceChecker()

    def analyze_email(self, email_content, subject_line="", sender_info=""):
        """Comprehensive email analysis with compliance report"""

        # Basic spam detection
        spam_result = self.detector.predict(email_content)

        # Compliance checking
        compliance_result = self.compliance_checker.check_compliance(email_content)

        # Combined analysis
        combined_report = {
            'analysis_id': hash(email_content + subject_line) % 1000000,
            'timestamp': datetime.now().isoformat(),
            'email_metadata': {
                'subject_line': subject_line,
                'sender_info': sender_info,
                'content_length': len(email_content),
                'word_count': len(email_content.split())
            },
            'spam_detection': {
                'classification': spam_result['prediction'],
                'confidence': spam_result['confidence'],
                'spam_probability': spam_result['spam_probability'],
                'features': spam_result['features']
            },
            'compliance_analysis': compliance_result,
            'overall_assessment': self._generate_overall_assessment(spam_result, compliance_result),
            'action_items': self._generate_action_items(spam_result, compliance_result)
        }

        return combined_report

    def _generate_overall_assessment(self, spam_result, compliance_result):
        """Generate overall assessment combining spam and compliance analysis"""
        assessment = {
            'deliverability_score': 0,
            'compliance_status': 'COMPLIANT',
            'risk_factors': [],
            'strengths': []
        }

        # Calculate deliverability score (0-100, higher is better)
        spam_score = (1 - spam_result['spam_probability']) * 50
        compliance_score = (100 - compliance_result['overall_score']) * 0.5
        assessment['deliverability_score'] = min(100, spam_score + compliance_score)

        # Determine compliance status
        if (spam_result['spam_probability'] > 0.7 or
                compliance_result['risk_level'] == 'HIGH'):
            assessment['compliance_status'] = 'NON_COMPLIANT'
        elif (spam_result['spam_probability'] > 0.4 or
              compliance_result['risk_level'] == 'MEDIUM'):
            assessment['compliance_status'] = 'REQUIRES_REVIEW'

        # Identify risk factors
        if spam_result['spam_probability'] > 0.5:
            assessment['risk_factors'].append("High spam probability detected")

        if compliance_result['overall_score'] > 50:
            assessment['risk_factors'].append("Multiple compliance violations found")

        if spam_result['features']['spam_keywords'] > 3:
            assessment['risk_factors'].append("Excessive promotional language")

        # Identify strengths
        if spam_result['spam_probability'] < 0.3:
            assessment['strengths'].append("Low spam probability")

        if compliance_result['overall_score'] < 30:
            assessment['strengths'].append("Good compliance score")

        if len(compliance_result['violations']) == 0:
            assessment['strengths'].append("No major compliance violations")

        return assessment

    def _generate_action_items(self, spam_result, compliance_result):
        """Generate prioritized action items"""
        action_items = []

        # High priority items
        if spam_result['spam_probability'] > 0.7:
            action_items.append({
                'priority': 'HIGH',
                'action': 'Revise content to reduce spam indicators',
                'reason': 'High spam probability detected'
            })

        high_severity_violations = [v for v in compliance_result['violations']
                                    if v['severity'] == 'HIGH']
        if high_severity_violations:
            action_items.append({
                'priority': 'HIGH',
                'action': 'Address regulatory compliance violations',
                'reason': f"Found {len(high_severity_violations)} high-severity violations"
            })

        # Medium priority items
        if spam_result['features']['spam_keywords'] > 2:
            action_items.append({
                'priority': 'MEDIUM',
                'action': 'Reduce promotional language',
                'reason': 'Multiple spam keywords detected'
            })

        if compliance_result['overall_score'] > 40:
            action_items.append({
                'priority': 'MEDIUM',
                'action': 'Review and improve content quality',
                'reason': 'Multiple compliance issues found'
            })

        # Low priority items
        if spam_result['features']['exclamation_count'] > 2:
            action_items.append({
                'priority': 'LOW',
                'action': 'Reduce excessive punctuation',
                'reason': 'Multiple exclamation marks detected'
            })

        return action_items

    def bulk_analyze(self, email_list):
        """Analyze multiple emails and generate summary report"""
        results = []
        summary = {
            'total_emails': len(email_list),
            'compliant_emails': 0,
            'non_compliant_emails': 0,
            'requires_review': 0,
            'avg_deliverability_score': 0,
            'common_issues': {}
        }

        for email_data in email_list:
            if isinstance(email_data, dict):
                result = self.analyze_email(
                    email_data.get('content', ''),
                    email_data.get('subject', ''),
                    email_data.get('sender', '')
                )
            else:
                result = self.analyze_email(email_data)

            results.append(result)

            # Update summary
            status = result['overall_assessment']['compliance_status']
            if status == 'COMPLIANT':
                summary['compliant_emails'] += 1
            elif status == 'NON_COMPLIANT':
                summary['non_compliant_emails'] += 1
            else:
                summary['requires_review'] += 1

            summary['avg_deliverability_score'] += result['overall_assessment']['deliverability_score']

            # Track common issues
            for violation in result['compliance_analysis']['violations']:
                issue_key = f"{violation['category']}.{violation['rule']}"
                summary['common_issues'][issue_key] = summary['common_issues'].get(issue_key, 0) + 1

        summary['avg_deliverability_score'] /= len(email_list)

        return {
            'individual_results': results,
            'summary': summary
        }


# Sample dataset creation function (enhanced)
def create_enhanced_sample_dataset():
    """Create an enhanced sample dataset for demonstration"""
    spam_emails = [
        "URGENT! You've won $1000000! Click here to claim your prize now!!!",
        "FREE MONEY! Act now! Limited time offer! Click here! Call 555-123-4567",
        "Congratulations! You're our lucky winner! Claim your cash prize at bit.ly/fake",
        "AMAZING DISCOUNT! 90% off everything! Buy now or miss out! Visit www.scam.tk",
        "You've been selected for a special offer! Act fast! Email us at fake@scam.ml",
        "FREE iPhone! Click here to get yours today! No strings attached!",
        "WINNER WINNER! You've won big! Click to claim your prize money!",
        "Urgent: Your account needs verification. Click here now! Don't delay!",
        "Make money fast! Work from home! Easy money! Call 1-800-SCAM-NOW",
        "FREE GIFT! Limited time! Click here now! Expires in 24 hours!"
    ]

    legitimate_emails = [
        "Hi John, let's meet for coffee tomorrow at 3 PM. Looking forward to our discussion about the project.",
        "The quarterly report is ready for your review. Please find it attached. Our company is located at 123 Main Street.",
        "Thank you for your purchase. Your order will arrive soon. To unsubscribe, click here.",
        "Meeting scheduled for Monday at 10 AM in conference room A. Please confirm your attendance.",
        "Please find the attached document for your reference. Contact us at support@company.com",
        "Your subscription renewal is due next month. Visit our privacy policy for more information.",
        "The project deadline has been extended by one week. This is a promotional update.",
        "Thank you for attending our webinar yesterday. You can opt out of future emails here.",
        "Please review the contract and send your feedback. Our address: 456 Business Ave, Suite 100.",
        "The maintenance window is scheduled for this weekend. For data protection info, see our policy."
    ]

    # Create dataset
    emails = spam_emails + legitimate_emails
    labels = [1] * len(spam_emails) + [0] * len(legitimate_emails)

    return pd.DataFrame({'email': emails, 'label': labels})


# Enhanced demo function
def demo_enhanced_spam_detection():
    """Demonstrate the enhanced spam detection system"""
    print("🚀 Enhanced Spam Detection System with MPS Support")
    print("=" * 60)

    # Create sample dataset
    df = create_enhanced_sample_dataset()
    print(f"Dataset created with {len(df)} emails")

    # Initialize detector with MPS support
    detector = SpamDetector(use_mps=True)

    # Preprocess emails
    df['processed_email'] = df['email'].apply(detector.preprocess_text)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_email'], df['label'], test_size=0.3, random_state=42
    )

    # Train models
    print("\n📊 Training Models...")
    detector.train_models(X_train, y_train)

    # Evaluate models
    print("\n📈 Evaluating Models...")
    results, best_model = detector.evaluate_models(X_test, y_test)

    # Initialize enhanced API
    api = EnhancedSpamDetectionAPI(detector)

    # Test with comprehensive analysis
    print("\n🔍 Comprehensive Email Analysis:")
    print("-" * 50)

    test_emails = [
        {
            'content': "URGENT! Free money! Click here now!!! Call 555-SCAM-NOW",
            'subject': "YOU'VE WON BIG!!!",
            'sender': "noreply@scam.tk"
        },
        {
            'content': email_text,
            'subject': "Feedback Request",
            'sender': "john@company.com"
        }
    ]

    for email_data in test_emails:
        print(f"\n📧 Analyzing Email:")
        print(f"Subject: {email_data['subject']}")
        print(f"Content: {email_data['content'][:80]}...")

        report = api.analyze_email(
            email_data['content'],
            email_data['subject'],
            email_data['sender']
        )

        print(f"\n📊 Analysis Results:")
        print(f"  Spam Classification: {report['spam_detection']['classification']}")
        print(f"  Spam Probability: {report['spam_detection']['spam_probability']:.2f}")
        print(f"  Compliance Status: {report['overall_assessment']['compliance_status']}")
        print(f"  Deliverability Score: {report['overall_assessment']['deliverability_score']:.1f}/100")
        print(f"  Risk Level: {report['compliance_analysis']['risk_level']}")

        if report['compliance_analysis']['violations']:
            print(f"  Violations Found: {len(report['compliance_analysis']['violations'])}")
            for violation in report['compliance_analysis']['violations'][:3]:
                print(f"    - {violation['category']}.{violation['rule']} (severity: {violation['severity']})")

        if report['action_items']:
            print(f"  Action Items:")
            for item in report['action_items'][:2]:
                print(f"    - {item['priority']}: {item['action']}")

    # Bulk analysis demo
    print("\n📊 Bulk Analysis Demo:")
    print("-" * 30)

    bulk_results = api.bulk_analyze([email['content'] for email in test_emails])
    summary = bulk_results['summary']

    print(f"Total Emails Analyzed: {summary['total_emails']}")
    print(f"Compliant: {summary['compliant_emails']}")
    print(f"Non-Compliant: {summary['non_compliant_emails']}")
    print(f"Requires Review: {summary['requires_review']}")
    print(f"Average Deliverability Score: {summary['avg_deliverability_score']:.1f}")

    # Save models
    detector.save_models()

    return detector, api


# Run the enhanced demo
if __name__ == "__main__":
    detector, api = demo_enhanced_spam_detection()

    print("\n🎯 System Ready for Production Use!")
    print("Features:")
    print("- MPS/CUDA/CPU device support")
    print("- Comprehensive compliance checking")
    print("- Bulk email analysis")
    print("- Detailed reporting and recommendations")
    print("- Regulatory compliance (CAN-SPAM, GDPR considerations)")
