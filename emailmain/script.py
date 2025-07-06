# Standard library imports
import re  # Regular expressions for text processing
import logging  # Logging functionality
from typing import Optional, Tuple  # Type hints
import warnings  # Warning management

# NLP and spell checking libraries
import spacy  # Core NLP processing
from spacy.tokens import Token  # Token extension for spaCy
import enchant  # PyEnchant for dictionary-based spell checking
import contextualSpellCheck  # Contextual spell checking for spaCy
from textblob import TextBlob  # Alternative spell checking
from transformers import BertTokenizer, BertForMaskedLM  # BERT for context-aware corrections
import torch  # PyTorch for BERT model
import contractions  # For expanding contractions

# Suppress UserWarnings which are common with NLP libraries
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiMethodSpellChecker:
    """
    A comprehensive spell checker that uses multiple NLP methods for correction.

    This class combines several approaches to text correction:
    1. spaCy for NLP processing and named entity recognition
    2. PyEnchant for dictionary-based spell checking
    3. BERT for context-aware word prediction
    4. Contextual spell checking for improved accuracy
    5. TextBlob as a fallback spell checker
    6. Regular expressions for pattern-based corrections

    Note: This is an earlier version of the spell checker that was later
    enhanced in the EnhancedSpellChecker class.
    """

    def __init__(self):
        """
        Initialize the spell checker with all required components.
        """
        # Components to be initialized
        self.nlp = None  # spaCy NLP pipeline
        self.dictionary = None  # PyEnchant dictionary
        self.tokenizer = None  # BERT tokenizer
        self.model = None  # BERT model

        # Dictionary of brand names that should be properly capitalized
        # Used to ensure consistent capitalization of brand names
        self.brand_names = {
            "airbnb": "Airbnb",
            "google": "Google",
            "apple": "Apple",
            "microsoft": "Microsoft",
            "facebook": "Facebook",
            "amazon": "Amazon",
            "uber": "Uber",
            "lyft": "Lyft",
            "netflix": "Netflix",
            "spotify": "Spotify",
            "instagram": "Instagram",
            "twitter": "Twitter",
            "linkedin": "LinkedIn",
            "youtube": "YouTube",
            "medallia": "Medallia"
        }

        # Initialize all components
        self.initialize_all_components()

    def initialize_spacy_with_spellcheck(self) -> spacy.language.Language:
        """Initialize spaCy with ContextualSpellCheck or fallback to basic pipeline"""
        nlp = spacy.load("en_core_web_sm")

        # Register the extension attribute
        if not Token.has_extension("contextual_spellcheck_ignore"):
            Token.set_extension("contextual_spellcheck_ignore", default=False)

        # Add ContextualSpellCheck to the pipeline if available
        if contextualSpellCheck:
            try:
                contextualSpellCheck.add_to_pipe(nlp)
                logger.info("ContextualSpellCheck added to spaCy pipeline")
            except Exception as e:
                logger.warning(f"Failed to add ContextualSpellCheck: {str(e)}")
        else:
            logger.warning("contextualSpellCheck not installed")

        return nlp

    def initialize_dictionary(self) -> Optional[enchant.Dict]:
        """Initialize pyenchant dictionary or provide fallback"""
        if enchant:
            try:
                broker = enchant.Broker()
                d = broker.request_dict("en_US")
                logger.info("Enchant dictionary for en_US loaded successfully")
                return d
            except Exception as e:
                logger.warning(f"Failed to load Enchant dictionary: {str(e)}")
                return None
        else:
            logger.warning("pyenchant not installed")
            return None

    def initialize_bert_model(self) -> Tuple[Optional[BertTokenizer], Optional[BertForMaskedLM]]:
        """Initialize BERT model and tokenizer"""
        if BertTokenizer and BertForMaskedLM and torch:
            try:
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = BertForMaskedLM.from_pretrained("bert-base-uncased")
                model.eval()

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                logger.info(f"BERT model loaded on {device}")

                return tokenizer, model
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {str(e)}")
                return None, None
        else:
            logger.warning("transformers or torch not installed")
            return None, None

    def initialize_all_components(self):
        """Initialize all spell checking components"""
        logger.info("Initializing spell checking components...")
        self.nlp = self.initialize_spacy_with_spellcheck()
        self.dictionary = self.initialize_dictionary()
        self.tokenizer, self.model = self.initialize_bert_model()

    def expand_contractions(self, text: str) -> str:
        """Expand contractions using the contractions library"""
        if not contractions:
            return text

        try:
            # First handle some common cases that might not be in the library
            manual_contractions = {
                "itll": "it'll",
                "youll": "you'll",
                "well": "we'll",
                "theyll": "they'll",
                "thatll": "that'll",
                "wholl": "who'll",
                "whatll": "what'll",
            }

            # Apply manual contractions first
            for contraction, expansion in manual_contractions.items():
                text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)

            # Then use the contractions library
            expanded = contractions.fix(text)
            return expanded
        except Exception as e:
            logger.warning(f"Contractions expansion failed: {str(e)}")
            return text

    def capitalize_after_punctuation(self, text: str) -> str:
        """Capitalize words after sentence-ending punctuation"""
        # Pattern to match sentence-ending punctuation followed by whitespace and a word
        pattern = r'([.!?])\s+([a-z])'

        def capitalize_match(match):
            return match.group(1) + ' ' + match.group(2).upper()

        # Apply capitalization after punctuation
        result = re.sub(pattern, capitalize_match, text)

        # Capitalize the first word of the text if it's lowercase
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        return result

    def capitalize_brand_names(self, text: str) -> str:
        """Capitalize brand names according to the brand_names dictionary"""
        result = text

        for lowercase_brand, proper_brand in self.brand_names.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(lowercase_brand) + r'\b'
            result = re.sub(pattern, proper_brand, result, flags=re.IGNORECASE)

        return result

    def bert_correct_token(self, token, line: str) -> str:
        """Correct a single token using BERT by masking it and predicting"""
        if not self.tokenizer or not self.model:
            return token.text

        # Skip if already valid
        if self.dictionary and self.dictionary.check(token.text):
            return token.text

        # Skip proper nouns and brand names (start with capital letter)
        if token.text[0].isupper() and len(token.text) > 2:
            return token.text

        # Find token position in line
        words = line.split()
        token_idx = -1
        for i, word in enumerate(words):
            if word == token.text and token_idx == -1:
                token_idx = i
                break

        if token_idx == -1:
            return token.text

        # Create masked version
        words[token_idx] = "[MASK]"
        masked_text = " ".join(words)

        try:
            # Tokenize and predict
            inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Find the mask token position in the tokenized input
            mask_token_idx = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]
            if len(mask_token_idx) == 0:
                return token.text

            # Get top 5 predictions and choose the best one
            predictions = torch.topk(outputs.logits[0, mask_token_idx[0]], 5)

            for pred_id in predictions.indices:
                predicted_token = self.tokenizer.decode([pred_id]).strip()

                # Skip invalid predictions
                if (predicted_token.startswith("##") or
                        not predicted_token.isalpha() or
                        len(predicted_token) < 2):
                    continue

                # Check if prediction is reasonable (similar length, similar starting letter)
                if (abs(len(predicted_token) - len(token.text)) > 2 or
                        predicted_token[0].lower() != token.text[0].lower()):
                    continue

                # Check if prediction is a valid word
                if self.dictionary and self.dictionary.check(predicted_token):
                    # Preserve original capitalization
                    if token.text[0].isupper():
                        predicted_token = predicted_token.capitalize()
                    return predicted_token

            return token.text

        except Exception as e:
            logger.warning(f"BERT correction failed for '{token.text}': {str(e)}")
            return token.text

    def protect_specific_terms(self, doc):
        """Protect specific terms from being spellchecked using dictionary and NER"""
        # Protect named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"]:
                for token in ent:
                    token._.contextual_spellcheck_ignore = True
                    logger.debug(f"Protected entity: {token.text} ({ent.label_})")

        # Protect other specific patterns
        for token in doc:
            if token._.contextual_spellcheck_ignore:
                continue

            # Brand names (check against our dictionary)
            if token.text.lower() in self.brand_names:
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected brand name: {token.text}")

            # Valid dictionary words
            elif self.dictionary and self.dictionary.check(token.text):
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected valid word: {token.text}")

            # Abbreviations (2 uppercase letters)
            elif len(token.text) == 2 and token.text.isupper():
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected abbreviation: {token.text}")

            # Numbers or mixed alphanumeric
            elif any(c.isdigit() for c in token.text):
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected number: {token.text}")

            # Long uppercase words (likely acronyms)
            elif token.text.isupper() and len(token.text) > 3:
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected uppercase: {token.text}")

            # URLs and email patterns
            elif re.match(r'^https?://|@.*\.com$|\.com$', token.text, re.IGNORECASE):
                token._.contextual_spellcheck_ignore = True
                logger.debug(f"Protected URL/email: {token.text}")

        return doc

    def apply_regex_corrections(self, text: str) -> str:
        """Apply regex-based corrections for specific patterns"""
        corrections = [
            # Number + time units
            (r'\b(\d+)minutes\b', r'\1 minutes'),
            (r'\b(\d+)seconds\b', r'\1 seconds'),
            (r'\b(\d+)hours\b', r'\1 hours'),
            (r'\b(\d+)days\b', r'\1 days'),
            (r'\b(\d+)weeks\b', r'\1 weeks'),
            (r'\b(\d+)months\b', r'\1 months'),
            (r'\b(\d+)years\b', r'\1 years'),

            # Common word concatenations
            (r'\byouranswers\b', 'your answers'),
            (r'\byourexperience\b', 'your experience'),
            (r'\byourtravel\b', 'your travel'),
            (r'\byourreservation\b', 'your reservation'),
            (r'\byourhotel\b', 'your hotel'),
            (r'\byourtrip\b', 'your trip'),

            # Other common patterns
            (r'\bmakesure\b', 'make sure'),
            (r'\bthankyou\b', 'thank you'),
            (r'\bpleasecheck\b', 'please check'),
            (r'\bclickhere\b', 'click here'),
        ]

        for pattern, replacement in corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def spell_check_text(self, text: str) -> str:
        """Perform spell checking while preserving formatting"""
        # First, expand contractions
        text = self.expand_contractions(text)

        # Apply regex corrections to the entire text first
        text = self.apply_regex_corrections(text)

        # Handle special characters
        special_chars = {'•', '‣', '›', '→'}
        char_map = {char: f' __{ord(char)}__ ' for char in special_chars}

        # Replace special characters with placeholders
        for char, placeholder in char_map.items():
            text = text.replace(char, placeholder)

        # Replace middle dots with spaces
        text = text.replace('·', ' ')

        lines = text.split('\n')
        corrected_lines = []

        for line in lines:
            if line.strip():
                doc = self.nlp(line)
                doc = self.protect_specific_terms(doc)
                corrected_tokens = []

                for token in doc:
                    logger.debug(f"Token: {token.text}, Ignored: {token._.contextual_spellcheck_ignore}")

                    if token._.contextual_spellcheck_ignore:
                        corrected_tokens.append(token.text_with_ws)
                        continue

                    corrected = token.text

                    # Try contextual spell check first
                    if hasattr(token._, 'contextual_spellcheck_suggestion'):
                        suggestion = token._.contextual_spellcheck_suggestion
                        if suggestion and suggestion != token.text:
                            corrected = suggestion
                            logger.info(f"ContextualSpellCheck: {token.text} → {corrected}")

                    # Try BERT correction if no contextual suggestion
                    elif self.tokenizer and self.model and not self.dictionary.check(token.text):
                        bert_corrected = self.bert_correct_token(token, line)
                        if bert_corrected != token.text:
                            corrected = bert_corrected
                            logger.info(f"BERT correction: {token.text} → {corrected}")

                    # Try TextBlob correction
                    elif TextBlob and not self.dictionary.check(token.text):
                        try:
                            blob = TextBlob(token.text)
                            corrected_text = str(blob.correct())
                            if corrected_text != token.text:
                                corrected = corrected_text
                                logger.info(f"TextBlob correction: {token.text} → {corrected}")
                        except Exception as e:
                            logger.warning(f"TextBlob correction failed: {e}")

                    # Try pyenchant correction as last resort
                    elif self.dictionary and not self.dictionary.check(token.text):
                        suggestions = self.dictionary.suggest(token.text)
                        if suggestions:
                            corrected = suggestions[0]
                            logger.info(f"Enchant correction: {token.text} → {corrected}")

                    corrected_tokens.append(corrected + token.whitespace_)

                corrected_line = ''.join(corrected_tokens)
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(line)

        result = '\n'.join(corrected_lines)

        # Restore special characters
        for char, placeholder in char_map.items():
            result = result.replace(placeholder, char)

        # Apply capitalization fixes
        result = self.capitalize_brand_names(result)
        result = self.capitalize_after_punctuation(result)

        return result


def main():
    """
    Main function to demonstrate the MultiMethodSpellChecker.

    This function:
    1. Initializes the spell checker
    2. Processes a sample text with common errors
    3. Displays the corrected version

    The sample text includes various issues like:
    - Brand name capitalization issues (airbnb -> Airbnb)
    - Missing apostrophes (itll -> it'll)
    - Missing spaces (3minutes -> 3 minutes)
    - Concatenated words (youranswers -> your answers)
    """
    # Initialize the spell checker
    checker = MultiMethodSpellChecker()

    # Sample text with various spelling and formatting errors
    raw_text = """airbnb
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

    # Process the text with the spell checker
    corrected_text = checker.spell_check_text(raw_text)

    # Display the corrected text
    print("Corrected Text:")
    print(corrected_text)


if __name__ == "__main__":
    main()
