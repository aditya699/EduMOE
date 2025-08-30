Tokenization = the process of mapping raw text (characters) into tokens (discrete units, integers).

Formally, we start with an alphabet (Σ, e.g., characters like a–z) and map it into a vocabulary (Δ, tokens).

A tokenizer is a pair of mappings:

Encoder (τ): text → tokens

Decoder (κ): tokens → text

In practice:

Input string "unbelievable"

Encoder → [“un”, “believ”, “able”]

IDs → [1053, 8274, 2145]

Decoder maps back to "unbelievable".

🔹 Why is Tokenization Needed?

Neural networks need numbers → you can’t feed raw text.

Vocabulary efficiency → words are too many, characters are too small; subwords (BPE, Unigram, WordPiece) balance the tradeoff.

Open vocabulary → tokenization allows us to handle unseen words by breaking them into subwords.

🔹 Middle Ground: Subword Tokenization

Option 3: Subwords (BPE, WordPiece, Unigram)

"unbelievable" → ["un", "believ", "able"]

Vocabulary size = ~30k–50k (manageable).

Handles new words: “un+believ+able” instead of OOV.

Works across languages too.

This is why all modern LLMs use subword tokenization.

🟢 Step 2.2 — How Byte Pair Encoding (BPE) Works
🔹 Intuition

BPE builds a subword vocabulary by iteratively merging frequent pairs of symbols.

Starts at the character level (so it can encode anything).

Gradually merges common pairs → “subwords” like th, ing, unbeliev.

Stops when vocab reaches a fixed size (e.g., 30k).