from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC

CORPUS_PATH = "data/corpus.txt"
OUT_JSON = "tokenizer/relationship_bpe.json"


import os
os.makedirs("tokenizer", exist_ok=True)

# ByteLevel BPE
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<URL>", "<USER>"]

trainer = BpeTrainer(
    vocab_size=16000,       
    min_frequency=2,
    special_tokens=special_tokens,
)

tokenizer.train([CORPUS_PATH], trainer)

tokenizer.save(OUT_JSON)
print("Saved tokenizer to:", OUT_JSON)


enc = tokenizer.encode("I think you should talk to your partner about this. <URL>")
print(enc.tokens)
print(enc.ids[:20])