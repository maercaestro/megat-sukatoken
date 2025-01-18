from transformers import AutoTokenizer
import tiktoken
from datasets import load_dataset
from sukatoken import MegatSukaToken  # Import your tokenizer class

# Initialize Tokenizers
suku_kata_tokenizer = MegatSukaToken()
bpe_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # Example with BERT
tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")  # Example with OpenAI's tokenizer

# Load the Dataset
print("Loading the dataset...")
dataset = load_dataset("maercaestro/pantun", split="train")
texts = dataset["text"]  # Extract the text column

# Define Tokenization Comparison Function
def compare_tokenization(texts, sample_size=100):
    """
    Compare the tokenization of MegatSukaToken, BPE, and tiktoken on a subset of texts.
    Args:
        texts (list): List of text samples.
        sample_size (int): Number of samples to test.

    Returns:
        dict: Dictionary containing tokenization results and vocabulary sizes.
    """
    results = []
    suku_kata_vocab = set()
    bpe_vocab = set()
    tiktoken_vocab = set()

    print(f"Comparing tokenization for {sample_size} samples...")
    for text in texts[:sample_size]:
        # Suku Kata Tokenizer
        suku_kata_tokens = suku_kata_tokenizer.tokenize(text)
        suku_kata_vocab.update(suku_kata_tokens)

        # BPE Tokenizer
        bpe_tokens = bpe_tokenizer.tokenize(text)
        bpe_vocab.update(bpe_tokens)

        # Tiktoken Tokenizer
        tiktoken_tokens = tiktoken_tokenizer.encode(text)
        tiktoken_vocab.update(tiktoken_tokens)

        # Append results
        results.append({
            "text": text,
            "suku_kata_tokens": suku_kata_tokens,
            "suku_kata_token_count": len(suku_kata_tokens),
            "bpe_tokens": bpe_tokens,
            "bpe_token_count": len(bpe_tokens),
            "tiktoken_tokens": tiktoken_tokens,
            "tiktoken_token_count": len(tiktoken_tokens)
        })

    # Calculate vocabulary sizes
    vocab_sizes = {
        "suku_kata_vocab_size": len(suku_kata_vocab),
        "bpe_vocab_size": len(bpe_vocab),
        "tiktoken_vocab_size": len(tiktoken_vocab)
    }

    return results, vocab_sizes

# Perform the Comparison
comparison_results, vocab_sizes = compare_tokenization(texts, sample_size=100)

# Calculate Percentage Differences
suku_kata_vocab_size = vocab_sizes["suku_kata_vocab_size"]
bpe_vocab_size = vocab_sizes["bpe_vocab_size"]
tiktoken_vocab_size = vocab_sizes["tiktoken_vocab_size"]

bpe_percentage = ((bpe_vocab_size - suku_kata_vocab_size) / suku_kata_vocab_size) * 100
tiktoken_percentage = ((tiktoken_vocab_size - suku_kata_vocab_size) / suku_kata_vocab_size) * 100

# Print Vocabulary Comparison
print("\nVocabulary Sizes:")
print(f"Suku Kata Vocabulary Size: {suku_kata_vocab_size}")
print(f"BPE Vocabulary Size: {bpe_vocab_size} ({bpe_percentage:+.2f}% vs. Suku Kata)")
print(f"Tiktoken Vocabulary Size: {tiktoken_vocab_size} ({tiktoken_percentage:+.2f}% vs. Suku Kata)")

# Print Results for First Few Samples
for idx, result in enumerate(comparison_results[:5]):  # Print first 5 results for brevity
    print(f"\nSample {idx + 1}:")
    print(f"Text: {result['text']}")
    print(f"Suku Kata Tokens: {result['suku_kata_tokens']} (Count: {result['suku_kata_token_count']})")
    print(f"BPE Tokens: {result['bpe_tokens']} (Count: {result['bpe_token_count']})")
    print(f"Tiktoken Tokens: {result['tiktoken_tokens']} (Count: {result['tiktoken_token_count']})")
