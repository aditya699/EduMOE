from typing import List
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer setup."""
    tokenizer_name: str = Field(default="gpt2", description="Pretrained tokenizer to use.")
    add_special_tokens: bool = Field(default=False, description="Whether to add special tokens like BOS/EOS.")


class TokenizerWrapper:
    """Wrapper for HuggingFace GPT-2 tokenizer."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # GPT-2 has no pad token by default → we assign EOS as pad for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> List[int]:
        """Convert text → token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=self.config.add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs → text."""
        return self.tokenizer.decode(tokens)

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size
        

# --- Usage ---
if __name__ == "__main__":
    config = TokenizerConfig(tokenizer_name="gpt2", add_special_tokens=False)
    tok = TokenizerWrapper(config)

    text = "EduMoE is our Mixture of Experts project."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)

    print("Vocab size:", tok.vocab_size())
    print("Text:", text)
    print("Tokens:", tokens)
    print("Decoded:", decoded)
