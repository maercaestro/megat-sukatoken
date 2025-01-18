from sukatoken import MegatSukaToken

# Initialize the tokenizer
tokenizer = MegatSukaToken()

# Sample dataset
dataset = [
    "kucing berlari di taman",
    "saya suka makan nasi lemak",
    "pohon besar tumbang semalam"
]

# Step 1: Build the vocabulary
vocab = tokenizer.build_vocab(dataset)
print("Vocabulary:", vocab)

vocab_file = "data/suku_kata_vocab.json"
tokenizer.save_vocab_to_file(vocab, vocab_file)
print(f"Vocabulary saved to {vocab_file}")

# Step 2: Tokenize a sentence
text = "pohon besar tumbang semalam"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Step 3: Encode the sentence into token IDs
encoded = tokenizer.encode_text(text, vocab)
print("Encoded Token IDs:", encoded)

# Step 4: Decode the token IDs back into text
decoded = tokenizer.decode_text(encoded, vocab)
print("Decoded Text:", decoded)
