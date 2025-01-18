# ________________________________________________
# Title:       MegatSukaToken (Malay Suku Kata Tokenizer)
# Created By:  Abu Huzaifah bin Haji Bidin
# Created On:  18 Jan 2025, with help of ChatGPT
# Purpose:     Syllable-based tokenizer specialized for Malay,
#              improved to handle multiple prefixes and preserve
#              morphological context using prefix markers.
# ________________________________________________

import re
import json
import string

class MegatSukaToken:
    """
    A tokenizer class that splits Malay text into suku kata (syllable units),
    with specialized handling for multiple Malay prefixes and common slang/particles.

    **Key Improvements**:
    1. Multiple Prefix Handling with Context Preservation:
       - Collect all prefixes in sequence, but do NOT remove them from the root.
       - Instead, generate prefix-marker tokens like "[PREFIX=mem]" and still keep the full word.

    2. Slang/Particle Splitting:
       - Splits out common Malay particles (e.g., 'lah', 'kan') if they appear as suffixes.

    3. Optional Punctuation Splitting:
       - If desired, punctuation can be isolated as separate tokens.

    4. More Extensive Docstrings & Comments
    """

    def __init__(self, split_punctuation=True):
        """
        Initializes the tokenizer with updated prefix and particle libraries.

        Args:
            split_punctuation (bool): If True, punctuation is split into separate tokens.
                                      Defaults to True.
        """
        self.prefixes = self._build_prefix_library()
        self.particles = self._build_particle_library()
        self.split_punctuation = split_punctuation

    @staticmethod
    def _build_prefix_library():
        """
        Builds a more extensive list of common Malay prefixes.
        Returns:
            list: A list of known prefixes, sorted from shorter to longer if needed.
        """
        return [
            "ber", "ter", "se", "be", "ke", "pe", "mem", "men", "meng", "meny",
            "pem", "pen", "peng", "peny", "penge", "pel", "per", "juru",
            "dwi", "eka", "pasca", "pra", "swa",
            # Keep old prefixes too:
            "ke", "pe", "pem", "pen", "peng", "penge", "pel", "per", "juru",
            "dwi", "eka", "pasca", "pra", "swa"
        ]

    @staticmethod
    def _build_particle_library():
        """
        Builds a list of common Malay slang/particles used at word endings.
        
        Returns:
            list: Slang words or discourse particles that may appear as suffixes.
        """
        return ["lah", "kan", "je", "pun", "tu", "keh", "dik", "tau", "nak"]

    def _split_prefixes_contextual(self, word):
        """
        Identifies all prefixes in sequence but retains the full word to preserve context.
        For each detected prefix, creates a marker token: "[PREFIX=<prefix>]".
        
        Example:
            "memperbaiki" -> ["[PREFIX=mem]", "[PREFIX=per]", "memperbaiki"]
            "menyapu" -> ["[PREFIX=meny]", "menyapu"]

        Args:
            word (str): A single word (in lowercase).

        Returns:
            list[str]: A list of prefix-marker tokens + the preserved word.
        """
        original_word = word
        prefixes_found = []

        while True:
            found_any = False
            for prefix in sorted(self.prefixes, key=len, reverse=True):
                if word.startswith(prefix):
                    prefixes_found.append(prefix)
                    word = word[len(prefix):]
                    found_any = True
                    break
            if not found_any:
                break

        tokens = []
        for pf in prefixes_found:
            tokens.append(f"[PREFIX={pf}]")

        # Keep the original word so the morphological structure remains intact
        tokens.append(original_word)
        return tokens

    def _split_particles(self, token):
        """
        Splits out a known particle if it appears at the end of the token.
        Example:
            'buatlah' -> ['buat', 'lah']
            'katakan' -> ['kata', 'kan']
        """
        for p in sorted(self.particles, key=len, reverse=True):
            # Only split if token ends with p AND it's not the entire token
            if token.endswith(p) and token != p:
                return [token[:-len(p)], p]
        return [token]

    def _tokenize_suku_kata(self, text):
        """
        Tokenizes a single word into Malay suku kata (syllables) using refined rules.

        1. A suku kata is a consonant followed by a vowel (CV).
        2. Syllable boundary is determined by a vowel or new CV pattern (CVC).
        3. If no CV pattern is found, treat the character individually.
        """
        tokens = []
        i = 0
        while i < len(text):
            if (i + 1 < len(text)
                and text[i] in "bcdfghjklmnpqrstvwxyz"
                and text[i + 1] in "aeiou"):
                start = i
                i += 2
                while i < len(text):
                    if text[i] in "aeiou":
                        break
                    elif (i + 1 < len(text)
                          and text[i] in "bcdfghjklmnpqrstvwxyz"
                          and text[i + 1] in "aeiou"):
                        break
                    else:
                        i += 1
                tokens.append(text[start:i])
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize(self, text):
        """
        Main tokenization flow:
          1. (Optional) Split punctuation.
          2. Split into words, lowercase each.
          3. For each word, produce prefix markers + preserved word.
          4. Tokenize that preserved word into suku kata.
          5. Attempt to split each suku kata token into particle suffix if matched.
        """
        if self.split_punctuation:
            for p in string.punctuation:
                text = text.replace(p, f" {p} ")

        words = text.split()
        tokens = []

        for word in words:
            w_lower = word.lower()
            prefix_and_word = self._split_prefixes_contextual(w_lower)

            for chunk in prefix_and_word:
                # If it's a prefix marker, add directly
                if chunk.startswith("[PREFIX="):
                    tokens.append(chunk)
                else:
                    # Otherwise, suku kata split & then particle check
                    suku_kata_tokens = self._tokenize_suku_kata(chunk)
                    for syl in suku_kata_tokens:
                        tokens.extend(self._split_particles(syl))

        return [t for t in tokens if t.strip()]

    def build_vocab(self, dataset):
        """
        Builds a fixed vocabulary from a list of texts.
        """
        vocab = set()
        for text in dataset:
            vocab.update(self.tokenize(text))

        vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        vocab["<UNK>"] = len(vocab)
        return vocab

    def encode_text(self, text, vocab):
        """
        Encodes a string into token IDs based on the fixed vocabulary.
        """
        tokens = self.tokenize(text)
        return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    def decode_text(self, token_ids, vocab):
        """
        Decodes token IDs back into text using the fixed vocabulary.
        """
        reverse_vocab = {idx: token for token, idx in vocab.items()}
        return " ".join(reverse_vocab.get(tid, "<UNK>") for tid in token_ids)

    def save_vocab_to_file(self, vocab, filepath):
        """
        Saves the vocabulary to a JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)


# Example usage:
# tokenizer = MegatSukaToken(split_punctuation=True)
# sample_text = "Memperbaiki kerosakanlah, kan?"
# tokens = tokenizer.tokenize(sample_text)
# print(tokens)  # Check the new prefix markers and suku kata + particle splits
