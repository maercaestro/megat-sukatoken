from transformers import AutoTokenizer
import tiktoken
from sukatoken import MegatSukaToken  # Import the class

# Initialize tokenizers
suku_kata_tokenizer = MegatSukaToken()
bpe_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # Example with BERT
tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")  # Example with OpenAI's tiktoken

# Malay sample words
malay_words = [
    "berlari", "kemungkinan", "kerusi", "pengkritik", "pemprosesan", "belajar", "sukarela", "menyelamatkan"
]

# Tokenization Comparison
def compare_tokenization(words):
    results = []
    for word in words:
        # Suku Kata Tokenizer
        suku_kata_tokens = suku_kata_tokenizer.tokenize(word)

        # BPE Tokenizer
        bpe_tokens = bpe_tokenizer.tokenize(word)

        # Tiktoken Tokenizer
        tiktoken_tokens = tiktoken_tokenizer.encode(word)

        # Append results
        results.append({
            "word": word,
            "suku_kata_tokens": suku_kata_tokens,
            "suku_kata_token_count": len(suku_kata_tokens),
            "bpe_tokens": bpe_tokens,
            "bpe_token_count": len(bpe_tokens),
            "tiktoken_tokens": tiktoken_tokens,
            "tiktoken_token_count": len(tiktoken_tokens)
        })

    return results

# Analyze and Print Results
results = compare_tokenization(malay_words)
for result in results:
    print(f"Word: {result['word']}")
    print(f"Suku Kata Tokens: {result['suku_kata_tokens']} (Count: {result['suku_kata_token_count']})")
    print(f"BPE Tokens: {result['bpe_tokens']} (Count: {result['bpe_token_count']})")
    print(f"Tiktoken Tokens: {result['tiktoken_tokens']} (Count: {result['tiktoken_token_count']})")
    print("---")
