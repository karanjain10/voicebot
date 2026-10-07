import re
import torch.nn as nn

def tokenize(text):
    words = re.findall(r"[a-z']+", text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]  # unigrams + bigrams

def bow(text, vocab):
    import numpy as np
    v = np.zeros(len(vocab), dtype="float32")
    for t in tokenize(text):
        if t in vocab:
            v[vocab[t]] = 1.0
    return v

class IntentNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_out))
    def forward(self, x):
        return self.net(x)
