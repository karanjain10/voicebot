import json, random
from datetime import datetime
import numpy as np, torch
from flask import Flask, jsonify, request, send_from_directory
from model import IntentNet, bow

CONF_MIN = 0.55  # below this, admit confusion instead of guessing

ck = torch.load("model.pt")
vocab, tags = ck["vocab"], ck["tags"]
net = IntentNet(len(vocab), len(tags)); net.load_state_dict(ck["state"]); net.eval()
responses = {i["tag"]: i["responses"] for i in json.load(open("data/intents.json"))["intents"]}

app = Flask(__name__, static_folder="static")

def reply(text):
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(bow(text, vocab))[None]), 1)[0]
    c, i = p.max(0)
    if c < CONF_MIN or not text.strip():
        return "unknown", float(c), "Sorry, I didn't understand that. Try asking for a joke, the time or my name."
    tag = tags[int(i)]
    r = random.choice(responses[tag])
    now = datetime.now()
    r = r.replace("__TIME__", now.strftime("It's %I:%M %p.")).replace("__DATE__", now.strftime("Today is %A, %d %B %Y."))
    return tag, float(c), r

@app.post("/chat")
def chat():
    text = (request.get_json(silent=True) or {}).get("text", "")[:300]
    tag, conf, r = reply(text)
    return jsonify(intent=tag, confidence=round(conf, 3), response=r)

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
