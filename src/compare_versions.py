import importlib.util
import sys

# Import the current version
from sukatoken import MegatSukaToken as CurrentToken

# Import the backup version
spec = importlib.util.spec_from_file_location("sukatoken_bak", "src/sukatoken_bak.py")
bak_module = importlib.util.module_from_spec(spec)
sys.modules["sukatoken_bak"] = bak_module
spec.loader.exec_module(bak_module)
BackupToken = bak_module.MegatSukaToken

# Initialize both tokenizers
current_tokenizer = CurrentToken()
backup_tokenizer = BackupToken()

# Test cases - complex Malay words with prefixes and particles
test_cases = [
    "memperbaiki",
    "pembelajaran",
    "keberhasilan",
    "menyelamatkan",
    "pengembangan",
    "berjalanlah",
    "membacakan",
    "persekolahan",
    "kebersamaannya",
    "mempermainkan"
]

print("Comparing Current vs Backup MegatSukaToken\n")
print("=" * 60)

for word in test_cases:
    print(f"\nWord: {word}")
    print("-" * 30)
    
    # Current version
    current_tokens = current_tokenizer.tokenize(word)
    print(f"Current: {current_tokens}")
    print(f"Token count: {len(current_tokens)}")
    
    # Backup version
    backup_tokens = backup_tokenizer.tokenize(word)
    print(f"Backup:  {backup_tokens}")
    print(f"Token count: {len(backup_tokens)}")
