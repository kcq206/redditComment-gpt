from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/relationship_bpe.json")
print(tokenizer.get_vocab_size())