import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

CORPUS_PATH = "data/corpus.txt"
TOKENIZER_PATH = "tokenizer/relationship_bpe.json"
OUT_DIR = "data/tokens"

os.makedirs(OUT_DIR, exist_ok=True)

tok = Tokenizer.from_file(TOKENIZER_PATH)

# If you trained with these special tokens, get their ids (optional but helpful)
special = {s: tok.token_to_id(s) for s in ["<bos>", "<eos>"] if tok.token_to_id(s) is not None}
BOS = special.get("<bos>")
EOS = special.get("<eos>")

all_ids = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Encoding lines"):
        line = line.strip()
        if not line:
            continue

        ids = tok.encode(line).ids

        # Optional: add BOS/EOS around each line
        if BOS is not None:
            all_ids.append(BOS)
        all_ids.extend(ids)
        if EOS is not None:
            all_ids.append(EOS)

arr = np.array(all_ids, dtype=np.uint16)  # uint16 supports vocab up to 65535
print("Total tokens:", arr.size)

# Save tokens to disk
out_path = os.path.join(OUT_DIR, "all_tokens.npy")
np.save(out_path, arr)
print("Saved:", out_path)