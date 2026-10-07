import json, random
import numpy as np, torch, torch.nn as nn
from model import IntentNet, bow, tokenize

random.seed(0); np.random.seed(0); torch.manual_seed(0)
intents = json.load(open("data/intents.json"))["intents"]
tags = [i["tag"] for i in intents]
samples = [(p, tags.index(i["tag"])) for i in intents for p in i["patterns"]]
vocab = {t: n for n, t in enumerate(sorted({t for p, _ in samples for t in tokenize(p)}))}

def fit(data, epochs=300):
    X = torch.tensor(np.stack([bow(p, vocab) for p, _ in data]))
    y = torch.tensor([l for _, l in data])
    m = IntentNet(len(vocab), len(tags))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        nn.functional.cross_entropy(m(X), y, label_smoothing=0.1).backward(); opt.step()
    return m.eval()

def acc(m, data):
    X = torch.tensor(np.stack([bow(p, vocab) for p, _ in data]))
    return (m(X).argmax(1).numpy() == np.array([l for _, l in data])).mean()

# stratified 75/25 split for honest evaluation
train, test = [], []
for t in range(len(tags)):
    s = [x for x in samples if x[1] == t]; random.shuffle(s)
    k = max(1, round(len(s) * 0.25)); test += s[:k]; train += s[k:]
m = fit(train)
print(f"vocab={len(vocab)} intents={len(tags)} samples={len(samples)}")
print(f"train acc={acc(m, train):.3f}  held-out test acc={acc(m, test):.3f} (n={len(test)})")

# ship a model trained on all data
m = fit(samples)
torch.save({"state": m.state_dict(), "vocab": vocab, "tags": tags}, "model.pt")
print("saved model.pt")
