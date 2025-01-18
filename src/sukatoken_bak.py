#________________________________
# Title:       MegatSukaToken (Malay Suku Kata Tokenizer)
# Created By:  Abu Huzaifah bin Haji Bidin
# Created On:  18 Jan 2025, with help of ChatGPT
# This code was made to tokenize Malay words into suku kata, with prefix handling. 
# This is to prove that tokenization using Malay syllabels produce a more meaningful tokenization compared to 
# industry standard tokenization methods (BPE, WordPiece, etc) for Malay language.
#________________________________________________________________________________

import re
import json

class MegatSukaToken:
    def __init__(self):
        self.prefixes = self.build_prefix_library()

    @staticmethod
    def build_prefix_library():
        """Builds a list of common Malay prefixes.Malay word has prefixes that added meanings to the words. This would 
        be helpful during attention mechanism in NLP tasks to capture context of the words."""
        return [
            "ke", "pe", "pem", "pen", "peng", "penge", "pel", "per", "juru",
            "dwi", "eka", "pasca", "pra", "swa"
        ]

    def split_prefix(self, word):
        """If the prefix exists in the words, split the prefix and the root word."""
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if word.startswith(prefix):
                return prefix, word[len(prefix):]
        return None, word

    def tokenize_suku_kata(self, text):
        """Tokenizes a word or sentence into Malay suku kata using refined rules. The rules are quite simple:
        1.  A suku kata is a consonant followed by a vowel (CV)
        2.  A suku kata boundary is determined by a vowel
        3.  Then next C after CV should be followed by V, if not, the C is added to the previous suku kata (CVC)
        3.  If no CV pattern is found, the character is added as is
        4.  Prefixes are handled separately
        """
        tokens = []
        i = 0
        while i < len(text):
            # Check for consonant followed by vowel (CV)
            if i + 1 < len(text) and text[i] in "bcdfghjklmnpqrstvwxyz" and text[i + 1] in "aeiou":
                # Start of a suku kata
                start = i
                i += 2  # Move past the CV

                # Look ahead to determine the suku kata boundary
                while i < len(text):
                    if text[i] in "aeiou":
                        # Found another vowel, stop the current token
                        break
                    elif i + 1 < len(text) and text[i] in "bcdfghjklmnpqrstvwxyz" and text[i + 1] in "aeiou":
                        # Found a new CV pattern, stop the current token
                        break
                    else:
                        # Continue adding consonants
                        i += 1

                # Add the token
                tokens.append(text[start:i])
            else:
                # If no CV pattern is found, add the character as is and move on
                tokens.append(text[i])
                i += 1

        return tokens

    def tokenize(self, text):
        """Main function to tokenize text with prefix handling."""
        words = text.split()
        tokens = []

        for word in words:
            prefix, root = self.split_prefix(word)
            if prefix:
                tokens.append(prefix)
            tokens.extend(self.tokenize_suku_kata(root))

        return tokens

    def build_vocab(self, dataset):
        """Builds a fixed vocabulary from a dataset."""
        vocab = set()
        for text in dataset:
            tokens = self.tokenize(text)
            vocab.update(tokens)
        vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        vocab["<UNK>"] = len(vocab)  # Add an unknown token
        return vocab

    def encode_text(self, text, vocab):
        """Encodes text into token IDs based on the fixed vocabulary."""
        tokens = self.tokenize(text)
        return [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    
    def decode_text(self, token_ids, vocab):
        """Decodes token IDs back into text using the fixed vocabulary."""
        reverse_vocab = {idx: token for token, idx in vocab.items()}
        tokens = [reverse_vocab.get(token_id, "<UNK>") for token_id in token_ids]
        return " ".join(tokens)

    def save_vocab_to_file(self, vocab, filepath):
        """Saves the vocabulary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
