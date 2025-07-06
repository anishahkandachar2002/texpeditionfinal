# import re
# import spacy
# from spacy.lang.en import English
# from wordsegment import load as segment_load, segment as segment_word
#
# # Load resources
# segment_load()
#
#
# # Build comprehensive dictionary from multiple sources
# def build_comprehensive_dictionary():
#     words = set()
#
#     # Option 1: Use spaCy's vocabulary (most comprehensive)
#     try:
#         import spacy
#         nlp = spacy.load("en_core_web_sm")
#         spacy_words = set()
#
#         # Extract words from spaCy's vocabulary
#         for word in nlp.vocab:
#             if word.is_alpha and len(word.text) > 1:  # Only alphabetic words, length > 1
#                 spacy_words.add(word.text.lower())
#
#         print(f"Loaded {len(spacy_words)} words from spaCy vocabulary")
#         words.update(spacy_words)
#
#     except Exception as e:
#         print(f"Could not load spaCy vocabulary: {e}")
#
#     # Option 2: Use SCOWL (Spell Checker Oriented Word Lists) via PyEnchant
#     try:
#         import enchant
#
#         # Get words from enchant dictionary
#         d = enchant.Dict("en_US")
#         # enchant doesn't have a direct word list, but we can use it for validation
#
#         # Common word patterns to validate
#         common_patterns = []
#
#         # Add common base words
#         base_words = [
#             'answer', 'question', 'minute', 'experience', 'guest', 'travel', 'plan',
#             'help', 'make', 'take', 'survey', 'team', 'policy', 'privacy', 'download',
#             'store', 'play', 'app', 'google', 'apple', 'unsubscribe', 'email',
#             'service', 'company', 'customer', 'user', 'account', 'profile', 'settings'
#         ]
#
#         for word in base_words:
#             if d.check(word):
#                 words.add(word)
#                 # Add common variations
#                 for suffix in ['s', 'ed', 'ing', 'er', 'est', 'ly']:
#                     variant = word + suffix
#                     if d.check(variant):
#                         words.add(variant)
#
#         print(f"Added words via PyEnchant validation")
#
#     except Exception as e:
#         print(f"PyEnchant not available: {e}")
#
#     # Option 3: Use WordNet from NLTK (more comprehensive than basic words)
#     try:
#         from nltk.corpus import wordnet
#         import nltk
#
#         wordnet_words = set()
#
#         # Get all words from WordNet
#         for synset in wordnet.all_synsets():
#             for lemma in synset.lemmas():
#                 word = lemma.name().replace('_', ' ')
#                 if word.isalpha():
#                     wordnet_words.add(word.lower())
#
#         print(f"Loaded {len(wordnet_words)} words from WordNet")
#         words.update(wordnet_words)
#
#     except Exception as e:
#         print(f"WordNet not available: {e}")
#
#     # Option 4: Use Brown Corpus (real-world usage)
#     try:
#         from nltk.corpus import brown
#         import nltk
#
#         try:
#             nltk.download('brown', quiet=True)
#         except:
#             pass
#
#         brown_words = set()
#         for word in brown.words():
#             if word.isalpha() and len(word) > 1:
#                 brown_words.add(word.lower())
#
#         print(f"Loaded {len(brown_words)} words from Brown Corpus")
#         words.update(brown_words)
#
#     except Exception as e:
#         print(f"Brown Corpus not available: {e}")
#
#     # Option 5: Fallback to NLTK words if others fail
#     try:
#         from nltk.corpus import words as nltk_words
#         import nltk
#
#         try:
#             nltk.download('words', quiet=True)
#         except:
#             pass
#
#         nltk_word_set = set(word.lower() for word in nltk_words.words())
#         print(f"Loaded {len(nltk_word_set)} words from NLTK words")
#         words.update(nltk_word_set)
#
#     except Exception as e:
#         print(f"NLTK words not available: {e}")
#
#     # Add contractions programmatically
#     contractions = generate_contractions(words)
#     words.update(contractions)
#
#     print(f"Total vocabulary size: {len(words)}")
#     return words
#
#
# def generate_contractions(base_words):
#     """Generate contractions from base words"""
#     contractions = set()
#
#     # Common contraction patterns
#     patterns = [
#         # 'll contractions
#         (['i', 'you', 'he', 'she', 'it', 'we', 'they', 'that', 'this'], 'll'),
#         # 've contractions
#         (['i', 'you', 'we', 'they'], 've'),
#         # 're contractions
#         (['you', 'we', 'they'], 're'),
#         # 'd contractions
#         (['i', 'you', 'he', 'she', 'it', 'we', 'they'], 'd'),
#         # 'm contractions
#         (['i'], 'm'),
#         # 's contractions
#         (['it', 'that', 'this', 'he', 'she'], 's'),
#     ]
#
#     for words_list, ending in patterns:
#         for word in words_list:
#             if word in base_words:
#                 contractions.add(word + "'" + ending)
#
#     # Negative contractions
#     negative_bases = ['do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
#                       'will', 'would', 'should', 'could', 'can', 'might', 'must', 'need', 'dare']
#
#     for base in negative_bases:
#         if base in base_words:
#             contractions.add(base + "n't")
#
#     # Special cases
#     special_contractions = ["can't", "won't", "shan't"]
#     contractions.update(special_contractions)
#
#     return contractions
#
#
# # Load the comprehensive dictionary
# ENGLISH_WORDS = build_comprehensive_dictionary()
#
# # Load spaCy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Please install the spaCy model: python -m spacy download en_core_web_sm")
#     nlp = English()
#     nlp.add_pipe("sentencizer")
#
# # Optional contextual spell check
# try:
#     import contextualSpellCheck
#
#     contextualSpellCheck.add_to_pipe(nlp)
#     nlp.contextual_spellCheck = True
#     SPELLCHECK_AVAILABLE = True
# except Exception as e:
#     print(f"Contextual spell check not available: {e}")
#     SPELLCHECK_AVAILABLE = False
#
#
# def try_apostrophe_insertion(word):
#     """Try inserting apostrophe at different positions to form valid dictionary words"""
#     word_lower = word.lower()
#
#     # If it's already a valid word, return as-is
#     if word_lower in ENGLISH_WORDS:
#         return word
#
#     # Try inserting apostrophe at each position
#     for i in range(1, len(word_lower)):
#         candidate = word_lower[:i] + "'" + word_lower[i:]
#         if candidate in ENGLISH_WORDS:
#             # Preserve original case
#             if word[0].isupper():
#                 return candidate.capitalize()
#             return candidate
#
#     return word
#
#
# def split_glued_words(text):
#     """Split glued words using wordsegment and spaCy, while avoiding duplicates."""
#     doc = nlp(text)
#     corrected = []
#
#     for token in doc:
#         word = token.text
#         whitespace = token.whitespace_
#         word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
#
#         # Skip punctuation, numbers, or spaces
#         if token.is_punct or token.like_num or token.is_space:
#             corrected.append(token.text_with_ws)
#             continue
#
#         # Preserve named entities and proper nouns
#         if token.ent_type_ or token.pos_ == "PROPN":
#             corrected.append(token.text_with_ws)
#             continue
#
#         # If it's a known word, don't try to split
#         if word_clean in ENGLISH_WORDS:
#             corrected.append(token.text_with_ws)
#             continue
#
#         # Strategy 1: Try apostrophe insertion (for any length word)
#         apostrophe_result = try_apostrophe_insertion(word)
#         if apostrophe_result != word:
#             corrected.append(apostrophe_result + whitespace)
#             continue
#
#         # Strategy 2: Try to split glued words
#         if len(word_clean) >= 3:
#             parts = segment_word(word_clean)
#
#             # Accept split only if:
#             # - we get multiple parts
#             # - all parts are valid dictionary words
#             # - the split is actually different from original
#             if (
#                     len(parts) > 1 and
#                     all(part in ENGLISH_WORDS for part in parts) and
#                     ' '.join(parts) != word_clean
#             ):
#                 corrected.append(' '.join(parts) + whitespace)
#                 continue
#
#         # If no corrections found, keep original
#         corrected.append(token.text_with_ws)
#
#     return ''.join(corrected).strip()
#
#
# def basic_cleanup(text):
#     text = text.replace('·', ' ').replace('\u00b7', ' ')
#     text = re.sub(r'([.!?,:;])([a-zA-Z])', r'\1 \2', text)
#     text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
#     text = re.sub(r'([a-z])([A-Z]{2,})', r'\1 \2', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()
#
#
# def clean_ocr_text_spacy(text, use_spellcheck=True):
#     lines = text.strip().split('\n')
#     cleaned_lines = []
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#
#         line = basic_cleanup(line)
#
#         try:
#             # First apply word splitting to the entire line
#             fixed_line = split_glued_words(line)
#
#             # Then apply spaCy processing for additional corrections
#             doc = nlp(fixed_line)
#             corrected = []
#
#             for token in doc:
#                 if token.is_punct or token.is_space:
#                     corrected.append(token.text_with_ws)
#                     continue
#
#                 if token.ent_type_ or token.pos_ == "PROPN":
#                     corrected.append(token.text_with_ws)
#                 elif (use_spellcheck and SPELLCHECK_AVAILABLE and
#                       hasattr(token._, 'suggestions') and token._.suggestions):
#                     suggestion = token._.suggestions[0]['text']
#                     corrected.append(suggestion + token.whitespace_)
#                 else:
#                     corrected.append(token.text_with_ws)
#
#             final_line = ''.join(corrected)
#
#         except Exception as e:
#             print(f"spaCy processing failed: {e}")
#             final_line = split_glued_words(line)
#
#         cleaned_lines.append(final_line)
#
#     cleaned_text = '\n'.join(cleaned_lines)
#     cleaned_text = re.sub(r'\n(?=[A-Z][a-z])', r'\n\n', cleaned_text)
#
#     return cleaned_text.strip()
#
#
# # === Example Usage ===
# if __name__ == "__main__":
#     # Test the specific sentence
#     test_sentence = "Itll only take 3 minutes, and youranswers will help us make Airbnb even"
#
#     print("Original:", test_sentence)
#     print("Cleaned: ", split_glued_words(test_sentence))
#
#     # Test individual words
#     print("\nTesting individual words:")
#     print("Itll ->", try_apostrophe_insertion("Itll"))
#     print("youranswers ->", split_glued_words("youranswers"))
#
#     # Debug: Check if words are in dictionary
#     print("\nDictionary check:")
#     print("'it'll' in dictionary:", "it'll" in ENGLISH_WORDS)
#     print("'your' in dictionary:", "your" in ENGLISH_WORDS)
#     print("'answers' in dictionary:", "answers" in ENGLISH_WORDS)
#
#     # Test full text
#     raw_text = """airbnb
# Hi Matthew,
# Thanks for using Airbnb. We really appreciate you choosing Airbnb for your travel
# plans.
# To help us improve, we'd like to ask you a few questions about your experience
# so far. Itll only take 3 minutes, and youranswers will help us make Airbnb even
# better for you and other guests.
# Thanks,
# The Airbnb Team
# Take the Survey
# 8+
# Sent with from Airbnb, Inc.
# 888 Brannan St, San Francisco, CA 94103
# Who is Medallia?·Unsubscribe·Privacy Policy
# DOWNLOAD ON
# DOWNLOAD ON
# App Store
# Google play"""
#
#     print("\n=== Cleaned OCR Text ===")
#     cleaned = clean_ocr_text_spacy(raw_text, use_spellcheck=False)
#     print(cleaned)