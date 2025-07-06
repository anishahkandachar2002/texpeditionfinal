# Standard library imports
import re  # Regular expressions for text processing
import logging  # Logging functionality
from typing import Optional, Tuple, Set, List  # Type hints
import warnings  # Warning management
import pickle  # For loading serialized data

# Third-party imports - spaCy for NLP processing
import spacy
from spacy.tokens import Token

# Load data from pickle file (faster than JSON)
# The data contains brand names, contractions, and other text processing resources
with open('text_processing_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Optional imports with fallbacks
# This approach allows the code to run with reduced functionality
# if some dependencies are not installed

# PyEnchant for spell checking
try:
    import enchant
except ImportError:
    enchant = None

# Contextual spell checking
try:
    import contextualSpellCheck
except ImportError:
    contextualSpellCheck = None

# TextBlob for additional spell checking
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# BERT for context-aware text correction
try:
    from transformers import BertTokenizer, BertForMaskedLM
    import torch
    import torch.nn.functional as F
except ImportError:
    BertTokenizer = None
    BertForMaskedLM = None
    torch = None
    F = None

# Contractions library for expanding contractions
try:
    import contractions
except ImportError:
    contractions = None

# Word segmentation for splitting concatenated words
try:
    from wordsegment import load as segment_load, segment as segment_word
    segment_load()
    WORDSEGMENT_AVAILABLE = True
except ImportError:
    WORDSEGMENT_AVAILABLE = False

# Suppress UserWarnings which are common with NLP libraries
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedSpellChecker:
    """
    A comprehensive spell checker that combines multiple NLP methods and approaches.
    Integrates the best features from both original implementations with BERT integration.

    This class provides text correction capabilities using:
    1. spaCy for NLP processing
    2. PyEnchant for dictionary-based spell checking
    3. BERT for context-aware word prediction
    4. Contextual spell checking for improved accuracy
    5. Word segmentation for fixing concatenated words
    6. Regular expressions for pattern-based corrections
    """

    def __init__(self, use_bert=True, use_spellcheck=True):
        """
        Initialize the spell checker with configurable components.

        Args:
            use_bert (bool): Whether to use BERT for context-aware corrections
            use_spellcheck (bool): Whether to use contextual spell checking
        """
        # Configuration flags
        self.use_bert = use_bert
        self.use_spellcheck = use_spellcheck

        # Components to be initialized
        self.nlp = None  # spaCy NLP pipeline
        self.dictionary = None  # PyEnchant dictionary
        self.tokenizer = None  # BERT tokenizer
        self.model = None  # BERT model
        self.device = None  # Computation device (CPU/GPU/MPS)
        self.english_words = set()  # Set of known English words

        # Load data from the imported resources
        self.brand_names = data["brand_names"]  # Brand name capitalization dictionary
        self.manual_contractions = data["manual_contractions"]  # Contraction mapping

        # Initialize all components
        self.initialize_all_components()

    def build_comprehensive_dictionary(self) -> Set[str]:
        """
        Build a comprehensive dictionary from multiple sources.

        This method combines words from:
        1. spaCy's vocabulary
        2. NLTK's word list
        3. WordNet's lemmas
        4. Generated contractions
        5. Common words from our data

        Returns:
            Set[str]: A set of known English words for spell checking
        """
        words = set()

        # Load from spaCy vocabulary
        if self.nlp:
            try:
                spacy_words = set()
                for word in self.nlp.vocab:
                    if word.is_alpha and len(word.text) > 1:
                        spacy_words.add(word.text.lower())
                logger.info(f"Loaded {len(spacy_words)} words from spaCy vocabulary")
                words.update(spacy_words)
            except Exception as e:
                logger.warning(f"Could not load spaCy vocabulary: {e}")

        # Try to load from NLTK if available
        try:
            from nltk.corpus import words as nltk_words
            import nltk
            try:
                # Download NLTK words corpus if not already present
                nltk.download('words', quiet=True)
                nltk_word_set = set(word.lower() for word in nltk_words.words())
                logger.info(f"Loaded {len(nltk_word_set)} words from NLTK")
                words.update(nltk_word_set)
            except:
                pass
        except ImportError:
            pass

        # Try to load from WordNet if available
        try:
            from nltk.corpus import wordnet
            wordnet_words = set()
            for synset in wordnet.all_synsets():
                for lemma in synset.lemmas():
                    word = lemma.name().replace('_', ' ')
                    if word.isalpha():
                        wordnet_words.add(word.lower())
            logger.info(f"Loaded {len(wordnet_words)} words from WordNet")
            words.update(wordnet_words)
        except:
            pass

        # Add contractions to the dictionary
        contractions = self.generate_contractions(words)
        words.update(contractions)

        # Add common words from our predefined list
        common_words = data['common_words']
        words.update(common_words)
        logger.info(f"Added {len(common_words)} common words")

        logger.info(f"Total vocabulary size: {len(words)}")
        return words

    def generate_contractions(self, base_words: Set[str]) -> Set[str]:
        """Generate contractions from base words"""
        contractions = set()

        # Common contraction patterns
        patterns = data['patterns']

        for words_list, ending in patterns:
            for word in words_list:
                if word in base_words:
                    contractions.add(word + "'" + ending)

        # Negative contractions
        negative_bases = data['negative_bases']

        for base in negative_bases:
            if base in base_words:
                contractions.add(base + "n't")

        # Special cases
        special_contractions = data['special_contractions']
        contractions.update(special_contractions)

        return contractions

    def initialize_spacy_with_spellcheck(self) -> spacy.language.Language:
        """Initialize spaCy with ContextualSpellCheck or fallback to basic pipeline"""
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic pipeline")
            from spacy.lang.en import English
            nlp = English()
            nlp.add_pipe("sentencizer")

        # Register the extension attribute
        if not Token.has_extension("contextual_spellcheck_ignore"):
            Token.set_extension("contextual_spellcheck_ignore", default=False)

        # Add ContextualSpellCheck to the pipeline if available and requested
        if contextualSpellCheck and self.use_spellcheck:
            try:
                contextualSpellCheck.add_to_pipe(nlp)
                logger.info("ContextualSpellCheck added to spaCy pipeline")
            except Exception as e:
                logger.warning(f"Failed to add ContextualSpellCheck: {str(e)}")

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
        return None

    def initialize_bert_model(self) -> Tuple[Optional[BertTokenizer], Optional[BertForMaskedLM]]:
        """Initialize BERT model and tokenizer"""
        if not self.use_bert or not BertTokenizer or not BertForMaskedLM or not torch:
            return None, None

        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
            model.eval()

            # Device selection with macOS Metal support
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA GPU")
            else:
                device = "cpu"
                logger.info("Using CPU")

            model = model.to(device)
            self.device = device
            logger.info(f"BERT model loaded on {device}")

            return tokenizer, model
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {str(e)}")
            return None, None

    def initialize_all_components(self):
        """Initialize all spell checking components"""
        logger.info("Initializing spell checking components...")
        self.nlp = self.initialize_spacy_with_spellcheck()
        self.dictionary = self.initialize_dictionary()
        self.tokenizer, self.model = self.initialize_bert_model()
        self.english_words = self.build_comprehensive_dictionary()

    def bert_correct_word(self, word: str, context: str, max_suggestions: int = 3) -> List[str]:
        """Use BERT to suggest corrections for a word in context"""
        if not self.use_bert or not self.tokenizer or not self.model:
            return []

        try:
            # Create masked sentence
            masked_sentence = context.replace(word, "[MASK]")

            # Tokenize
            inputs = self.tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Find mask token position
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

            if len(mask_token_index) == 0:
                return []

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            # Get top predictions for the masked token
            mask_token_logits = predictions[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, max_suggestions * 3, dim=1).indices[0].tolist()

            suggestions = []
            for token_id in top_tokens:
                token = self.tokenizer.decode([token_id]).strip()

                # Filter suggestions
                if (token.isalpha() and
                        len(token) > 1 and
                        token.lower() != word.lower() and
                        token not in suggestions):
                    suggestions.append(token)

                if len(suggestions) >= max_suggestions:
                    break

            return suggestions

        except Exception as e:
            logger.warning(f"BERT correction failed for '{word}': {str(e)}")
            return []

    def bert_correct_sentence(self, sentence: str, suspicious_words: List[str]) -> str:
        """Use BERT to correct multiple words in a sentence"""
        if not self.use_bert or not suspicious_words:
            return sentence

        corrected_sentence = sentence

        for word in suspicious_words:
            if word in corrected_sentence:
                suggestions = self.bert_correct_word(word, corrected_sentence)
                if suggestions:
                    # Use the top suggestion
                    best_suggestion = suggestions[0]
                    # Preserve original case
                    if word[0].isupper():
                        best_suggestion = best_suggestion.capitalize()
                    corrected_sentence = corrected_sentence.replace(word, best_suggestion, 1)

        return corrected_sentence

    def is_likely_misspelled(self, word: str) -> bool:
        """Check if a word is likely misspelled"""
        word_lower = word.lower()

        # Check if it's a known good word
        if word_lower in self.english_words:
            return False

        # Check with dictionary
        if self.dictionary and self.dictionary.check(word):
            return False

        # Check if it's a brand name
        if word_lower in self.brand_names:
            return False

        # Check if it's a number or contains numbers
        if any(c.isdigit() for c in word):
            return False

        # Check if it's an acronym
        if word.isupper() and len(word) > 1:
            return False

        # Check if it's too short
        if len(word) < 3:
            return False

        return True

    def expand_contractions(self, text: str) -> str:
        """Expand contractions using manual patterns and contractions library"""
        # First handle manual contractions
        for contraction, expansion in self.manual_contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)

        # Then use the contractions library if available
        if contractions:
            try:
                expanded = contractions.fix(text)
                return expanded
            except Exception as e:
                logger.warning(f"Contractions expansion failed: {str(e)}")
                return text

        return text

    def try_apostrophe_insertion(self, word: str) -> str:
        """Try inserting apostrophe at different positions to form valid words"""
        word_lower = word.lower()

        # If it's already a valid word, return as-is
        if word_lower in self.english_words:
            return word

        # Try inserting apostrophe at each position
        for i in range(1, len(word_lower)):
            candidate = word_lower[:i] + "'" + word_lower[i:]
            if candidate in self.english_words:
                # Preserve original case
                if word[0].isupper():
                    return candidate.capitalize()
                return candidate

        return word

    def split_glued_words(self, word: str) -> str:
        """Split glued words using wordsegment"""
        if not WORDSEGMENT_AVAILABLE:
            return word

        word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()

        # If it's already a known word, don't split
        if word_clean in self.english_words:
            return word

        # Try to split
        if len(word_clean) >= 3:
            parts = segment_word(word_clean)

            # Accept split only if we get multiple parts and all are valid
            if (len(parts) > 1 and
                    all(part in self.english_words for part in parts) and
                    ' '.join(parts) != word_clean):
                return ' '.join(parts)

        return word

    def apply_regex_corrections(self, text: str) -> str:
        """Apply regex-based corrections for specific patterns"""
        corrections = data['corrections']

        for pattern, replacement in corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def basic_cleanup(self, text: str) -> str:
        """Basic text cleanup - minimal changes to preserve structure"""
        # Fix spacing around punctuation (but preserve line structure)
        text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)

        # Fix spacing between numbers and letters
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)

        # Fix spacing between lowercase and uppercase sequences
        text = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', text)

        # Normalize whitespace but preserve single spaces and structure
        text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces and tabs, not newlines

        return text

    def capitalize_after_punctuation(self, text: str) -> str:
        """Capitalize words after sentence-ending punctuation while preserving structure"""
        lines = text.split('\n')
        result_lines = []

        for line in lines:
            if not line.strip():
                result_lines.append(line)
                continue

            # Pattern to match sentence-ending punctuation followed by whitespace and a word
            pattern = r'([.!?])\s+([a-z])'

            def capitalize_match(match):
                return match.group(1) + ' ' + match.group(2).upper()

            # Apply capitalization after punctuation
            result_line = re.sub(pattern, capitalize_match, line)

            # Capitalize the first word of the line if it's lowercase
            if result_line and result_line[0].islower():
                result_line = result_line[0].upper() + result_line[1:]

            result_lines.append(result_line)

        return '\n'.join(result_lines)

    def capitalize_brand_names(self, text: str) -> str:
        """Capitalize brand names according to the brand_names dictionary"""
        result = text

        for lowercase_brand, proper_brand in self.brand_names.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(lowercase_brand) + r'\b'
            result = re.sub(pattern, proper_brand, result, flags=re.IGNORECASE)

        return result

    def protect_specific_terms(self, doc):
        """Protect specific terms from being spellchecked"""
        # Protect named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"]:
                for token in ent:
                    token._.contextual_spellcheck_ignore = True

        # Protect other specific patterns
        for token in doc:
            if token._.contextual_spellcheck_ignore:
                continue

            # Brand names
            if token.text.lower() in self.brand_names:
                token._.contextual_spellcheck_ignore = True

            # Valid dictionary words
            elif (self.dictionary and self.dictionary.check(token.text)) or token.text.lower() in self.english_words:
                token._.contextual_spellcheck_ignore = True

            # Abbreviations
            elif len(token.text) == 2 and token.text.isupper():
                token._.contextual_spellcheck_ignore = True

            # Numbers or mixed alphanumeric
            elif any(c.isdigit() for c in token.text):
                token._.contextual_spellcheck_ignore = True

            # Long uppercase words (likely acronyms)
            elif token.text.isupper() and len(token.text) > 3:
                token._.contextual_spellcheck_ignore = True

            # URLs and email patterns
            elif re.match(r'^https?://|@.*\.com$|\.com$', token.text, re.IGNORECASE):
                token._.contextual_spellcheck_ignore = True

        return doc

    def correct_text(self, text: str) -> str:
        """
        Main text correction method that preserves original structure.

        This method applies a series of text corrections while maintaining
        the original formatting, line breaks, and structure of the text.

        The correction process includes:
        1. Expanding contractions
        2. Applying regex-based corrections
        3. Preserving special characters
        4. Line-by-line processing with spaCy
        5. BERT-based contextual corrections for suspicious words
        6. Token-by-token corrections with multiple fallback methods
        7. Final capitalization and formatting fixes

        Args:
            text (str): The input text to correct

        Returns:
            str: The corrected text with preserved structure
        """
        # First, expand contractions (e.g., "don't" -> "do not")
        text = self.expand_contractions(text)

        # Apply regex corrections to the entire text first (fixes common patterns)
        text = self.apply_regex_corrections(text)

        # Handle special characters by temporarily replacing them with placeholders
        # This prevents them from being lost or altered during processing
        special_chars = {'•', '‣', '›', '→'}
        char_map = {char: f' __{ord(char)}__ ' for char in special_chars}

        # Replace special characters with placeholders
        for char, placeholder in char_map.items():
            text = text.replace(char, placeholder)

        # Replace middle dots with spaces but preserve structure
        text = text.replace('·', ' ').replace('\u00b7', ' ')

        # Split into lines and process each line individually to preserve structure
        lines = text.split('\n')
        corrected_lines = []

        for line in lines:
            # Preserve empty lines exactly as they are
            if not line.strip():
                corrected_lines.append(line)
                continue

            # Apply minimal cleanup only to non-empty lines
            cleaned_line = self.basic_cleanup(line)

            # Process with spaCy for linguistic analysis
            doc = self.nlp(cleaned_line)
            doc = self.protect_specific_terms(doc)  # Prevent correction of valid terms

            # Collect suspicious words for BERT correction
            suspicious_words = []
            for token in doc:
                if (not token.is_punct and
                        not token.is_space and
                        not token._.contextual_spellcheck_ignore and
                        self.is_likely_misspelled(token.text)):
                    suspicious_words.append(token.text)

            # Use BERT for sentence-level correction if we have suspicious words
            # BERT provides context-aware corrections
            if suspicious_words and self.use_bert:
                corrected_line = self.bert_correct_sentence(cleaned_line, suspicious_words)
            else:
                # Fall back to token-by-token correction when BERT is unavailable
                corrected_tokens = []
                for token in doc:
                    # Preserve punctuation and whitespace
                    if token.is_punct or token.is_space:
                        corrected_tokens.append(token.text_with_ws)
                        continue

                    # Skip tokens marked to ignore
                    if token._.contextual_spellcheck_ignore:
                        corrected_tokens.append(token.text_with_ws)
                        continue

                    corrected = token.text

                    # Try multiple correction methods in sequence:

                    # 1. Try apostrophe insertion (e.g., "dont" -> "don't")
                    apostrophe_result = self.try_apostrophe_insertion(token.text)
                    if apostrophe_result != token.text:
                        corrected = apostrophe_result

                    # 2. Try word splitting (e.g., "helloworld" -> "hello world")
                    elif WORDSEGMENT_AVAILABLE:
                        split_result = self.split_glued_words(token.text)
                        if split_result != token.text:
                            corrected = split_result

                    # 3. Try contextual spell check
                    elif (self.use_spellcheck and
                          hasattr(token._, 'contextual_spellcheck_suggestion') and
                          token._.contextual_spellcheck_suggestion):
                        corrected = token._.contextual_spellcheck_suggestion

                    # 4. Try TextBlob as fallback
                    elif TextBlob and not token.text.lower() in self.english_words:
                        try:
                            blob = TextBlob(token.text)
                            corrected_text = str(blob.correct())
                            if corrected_text != token.text:
                                corrected = corrected_text
                        except Exception:
                            pass

                    corrected_tokens.append(corrected + token.whitespace_)

                corrected_line = ''.join(corrected_tokens)

            corrected_lines.append(corrected_line)

        # Rejoin the corrected lines, preserving original line breaks
        result = '\n'.join(corrected_lines)

        # Restore special characters from placeholders
        for char, placeholder in char_map.items():
            result = result.replace(placeholder, char)

        # Apply final capitalizations and formatting
        result = self.capitalize_brand_names(result)  # Fix brand name capitalization
        result = self.capitalize_after_punctuation(result)  # Fix sentence capitalization

        return result


def main():
    """
    Main function to demonstrate the spell checker.

    This function:
    1. Initializes the EnhancedSpellChecker with BERT and contextual spell checking
    2. Processes a sample text with common errors
    3. Displays the original and corrected versions

    The sample text includes:
    - Brand name capitalization issues
    - Missing apostrophes in contractions
    - Concatenated words
    - Missing spaces after numbers
    - Formatting issues
    """
    # Initialize the spell checker with all features enabled
    checker = EnhancedSpellChecker(use_bert=True, use_spellcheck=True)

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

    # Display the original text
    print("Original Text:")
    print(raw_text)
    print("\n" + "=" * 50 + "\n")

    # Process the text with the spell checker
    corrected_text = checker.correct_text(raw_text)

    # Display the corrected text
    print("Corrected Text:")
    print(corrected_text)


if __name__ == "__main__":
    main()
