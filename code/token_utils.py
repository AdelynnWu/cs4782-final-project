from transformers import CLIPTokenizer
from common_words import _COMMON_WORDS


def find_rare_tokens(
    tokenizer_name: str = "openai/clip-vit-large-patch14",
    num_candidates: int = 20,
    token_range: tuple[int, int] = (5000, 10000),
    max_char_length: int = 3,
) -> list[dict]:
    """
    Find rare tokens in the CLIP vocabulary that are good identifier candidates.

    Args:
    tokenizer_name: HuggingFace tokenizer to search
    num_candidates: How many candidate tokens to return
    token_range: Range of token IDs to search (less common tokens)
    max_char_length: Maximum character length for decoded token

    Returns:
    List of dicts with token_id, token_str, and decoded text
    """
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    candidates = []
    start, end = token_range
    end = min(end, tokenizer.vocab_size)

    for token_id in range(start, end):
        decoded = tokenizer.decode([token_id], skip_special_tokens=True).strip()

        if not decoded or " " in decoded or len(decoded) > max_char_length:
            continue

        if decoded.lower() in _COMMON_WORDS:
            continue

        re_tokenized = tokenizer.encode(decoded, add_special_tokens=False)
        if len(re_tokenized) == 1 and re_tokenized[0] == token_id:
            candidates.append(
                {
                    "token_id": token_id,
                    "token_str": decoded,
                    "vocab_index": token_id,
                }
            )

        if len(candidates) >= num_candidates:
            break
    return candidates


def print_candidates():
    print("=" * 60)
    print("DreamBooth Rare Token Identifier Candidates")
    print("=" * 60)
    print()
    print("These tokens have minimal semantic prior in the CLIP")
    print("tokenizer, making them good unique identifiers for binding")
    print("to a specific subject.")
    print()

    candidates = find_rare_tokens()

    print(f"{'Token ID':>10}  {'Token String':<15}")
    print("-" * 30)
    for c in candidates:
        print(f"{c['token_id']:>10}  {c['token_str']:<15}")

    print()
    print("Recommended: Pick one and use it as `identifier_token` in config.py")
    print("Default 'sks' (token 48136 in CLIP) is a commonly used choice.")
    print()
    print("Usage example prompt: 'a sks dog on the beach'")


if __name__ == "__main__":
    print_candidates()
