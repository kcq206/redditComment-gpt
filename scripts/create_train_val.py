import os
import numpy as np

TOKENS_PATH = "data/tokens/all_tokens.npy"
OUT_DIR = "data/tokens"
SEQ_LEN = 256          
VAL_FRACTION = 0.02    # 2% validation

tokens = np.load(TOKENS_PATH)
print("Total tokens:", len(tokens))

# Split by time/position (simple + fine)
split_idx = int(len(tokens) * (1 - VAL_FRACTION))
train = tokens[:split_idx]
val = tokens[split_idx:]

np.save(os.path.join(OUT_DIR, f"train_{SEQ_LEN}.npy"), train)
np.save(os.path.join(OUT_DIR, f"val_{SEQ_LEN}.npy"), val)

print("Train tokens:", len(train))
print("Val tokens:", len(val))
print("Saved train/val .npy")