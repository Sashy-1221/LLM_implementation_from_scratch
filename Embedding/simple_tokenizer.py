
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

vocab = {token: integer for integer, token in enumerate(preprocessed)}

class tokenizer():
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:val for i, val in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        return [self.str_to_int[i] for i in preprocessed]

    def decode(self,tokens):
        conv = [self.int_to_str[i] for i in tokens]
        return " ".join(conv)


obj = tokenizer(vocab)

text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = obj.encode(text= text)
print(ids)