import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- config ----
SEQ_LEN = 256
BATCH_SIZE = 16
EMBED = 384
LAYERS = 6
HEADS = 6
DROPOUT = 0.1
LR = 3e-4
STEPS = 3000
EVAL_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = f"data/tokens/train_{SEQ_LEN}.npy"
VAL_PATH   = f"data/tokens/val_{SEQ_LEN}.npy"


VOCAB_SIZE = 16000  

# ---- data ----
train_data = np.load(TRAIN_PATH)
val_data   = np.load(VAL_PATH)

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    # pick random starting points
    ix = np.random.randint(0, len(data) - SEQ_LEN - 1, size=(BATCH_SIZE,))
    x = np.stack([data[i:i+SEQ_LEN] for i in ix])
    y = np.stack([data[i+1:i+SEQ_LEN+1] for i in ix])
    x = torch.tensor(x, dtype=torch.long, device=DEVICE)
    y = torch.tensor(y, dtype=torch.long, device=DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ---- model ----
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=DROPOUT, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        # self-attention
        x_ln = self.ln1(x)
        # causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED)
        self.pos_emb = nn.Embedding(SEQ_LEN, EMBED)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block(EMBED, HEADS) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

model = TinyGPT().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

print("Device:", DEVICE)
print("Params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# ---- train loop ----
for step in range(1, STEPS + 1):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"step {step} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")

# save model
torch.save(model.state_dict(), "model_tinygpt.pt")
print("Saved model_tinygpt.pt")