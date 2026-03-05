import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer


SEQ_LEN = 256
EMBED = 384
LAYERS = 6
HEADS = 6
DROPOUT = 0.1
VOCAB_SIZE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model
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
        x_ln = self.ln1(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=mask, need_weights=False)
        x = x + attn_out
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

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x) 
        return logits

# ---- load tokenizer + model ----
tokenizer = Tokenizer.from_file("tokenizer/relationship_bpe.json")

model = TinyGPT().to(DEVICE)
state = torch.load("model_tinygpt.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

@torch.no_grad()
def generate(prompt: str, max_new_tokens=100, temperature=1.0, top_k=50):
    ids = tokenizer.encode(prompt).ids
    if len(ids) == 0:
        ids = [0]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None, :]  # (1, T)

    for _ in range(max_new_tokens):
        x_cond = x[:, -SEQ_LEN:]

        logits = model(x_cond)           # (1, T, V)
        logits = logits[:, -1, :]        # (1, V) last position

        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    response = tokenizer.decode(out_ids)
    response = response.replace("Ġ", " ")
    return response

if __name__ == "__main__":
    print("Device:", DEVICE)
    print(generate("i want to", max_new_tokens=80, temperature=0.9, top_k=50))