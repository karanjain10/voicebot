# Voice-Enabled Chatbot using Speech Recognition and Deep Learning

**Live link:** https://adiag1205-voicebot.static.hf.space  |  **Source:** https://github.com/karanjain10/voicebot

## 1. Objective
A publicly hosted chatbot that takes spoken input, converts it to text, classifies the user's intent with a neural network, and replies in text and synthesized speech. Both the recognized speech and the reply are shown on screen.

## 2. Dataset
Custom intent dataset (`data/intents.json`) written for this project: **15 intents** (greeting, goodbye, thanks, name, capabilities, joke, time, date, weather, motivation, study tips, how-are-you, deep learning, creator, music), **161 example utterances** (10–14 per intent), each intent with 1–3 canned responses. Time/date intents fill in the live value at reply time.
Preprocessing: lowercase, regex word tokenization, unigrams + bigrams, vocabulary of **496 features**, binary bag-of-words vectors.

## 3. Methodology
1. **Speech recognition:** browser Web Speech API (`SpeechRecognition`, en-US) streams microphone audio to the browser's built-in recognizer and returns a transcript; interim results are shown live. This is a pretrained deep-learning ASR system, so no server-side audio handling is needed.
2. **Intent classification:** transcript → `POST /chat` (Flask) → bag-of-words vector → PyTorch MLP → softmax over 15 intents.
3. **Response:** a random response for the predicted intent is returned. If the top softmax probability is below 0.55 the bot says it did not understand, rather than guessing.
4. **Voice output:** the reply is spoken with the browser's `speechSynthesis`.

## 4. Model architecture
`Linear(496→128) → ReLU → Dropout(0.4) → Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→15)`
Loss: cross-entropy. Optimizer: Adam, lr 3e-3, 300 full-batch epochs. Dropout is high because the dataset is small.

## 5. Results
Stratified 75/25 split (124 train / 37 test, seed 0):

| Set | Accuracy |
|---|---|
| Train | 100% |
| Held-out test | **81.1%** |

The shipped model is retrained on all 161 samples. Spot checks through the live API classify "hello there", "can you tell me a joke", "what time is it right now", "who built you" and "I feel like quitting" correctly with high confidence, and rejects out-of-domain input ("purple elephant bananas", confidence 0.45) with the fallback reply.

## 6. Limitations / future work
- Small hand-written dataset and a bag-of-words model: it fails on paraphrases that share no words with the training set. More data, or a pretrained embedding/transformer encoder, would improve it.
- ASR quality depends on the browser (Chrome/Edge/Safari) and sends audio to the browser vendor's service. Firefox does not support it; typed input is the fallback.
- No dialogue memory; each message is classified independently. The weather intent has no live data.

## 7. Deployment
Hosted free on a Hugging Face **static Space** (HTTPS, required by the microphone API): https://huggingface.co/spaces/adiag1205/voicebot. Free tiers do not offer a Python server, so `export_web.py` exports the trained PyTorch weights to `model.json` and the same forward pass runs in the browser in JavaScript (parity-checked against PyTorch on sample inputs). The Flask + Docker server version (`app.py`, `Dockerfile`, `render.yaml`) is kept in the repository for server-side hosting. First load of the site can take up to 1 minute.
