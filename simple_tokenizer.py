# simple_tokenizer.py

from transformers import BertTokenizer

class SimpleTokenizer(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('biobert-base-cased')

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

# ... other utility functions ...
