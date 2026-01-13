with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

vocab = {token: idx for idx, token in enumerate(preprocessed)}

obj = tokenizer(vocab)

text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = obj.encode(text= text)
print(ids)