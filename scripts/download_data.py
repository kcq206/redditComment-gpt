from datasets import load_dataset
import re
ds = load_dataset(
    "HuggingFaceGECLM/REDDIT_comments",
    split="relationship_advice",
    streaming = True
)

url_re = re.compile(r"https?://\S+")
ws_re = re.compile(r"\s+")
REMOVED = {"[deleted]", "[removed]", ""}

def clean(t: str):
    if not t:
        return None
    t = t.strip()
    if t.lower() in REMOVED:
        return None
    t = url_re.sub("<URL>", t)
    t = ws_re.sub(" ", t)
    if len(t) < 40:
        return None
    return t

MAX = 200_000
kept = 0

with open("data/corpus.txt", "w", encoding="utf-8") as f:
    for ex in ds:
        
        text = ex.get("selftext") or ex.get("text") or ex.get("body") or ""
        text = clean(text)
        if not text:
            continue
        f.write(text + "\n")
        kept += 1
        if kept >= MAX:
            break

print("wrote:", kept)