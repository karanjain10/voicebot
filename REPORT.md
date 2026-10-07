# Voice-Enabled Chatbot using Speech Recognition and Deep Learning

**Live link:** https://adiag1205-voicebot.static.hf.space  |  **Source:** https://github.com/karanjain10/voicebot

## 1. Objective
A publicly hosted chatbot that takes spoken input, converts it to text, classifies the user's intent with a neural network, and replies in text and synthesized speech. Both the recognized speech and the reply are shown on screen.

## 2. Dataset
Custom intent dataset (`data/intents.json`) written for this project: **21 intents** (greeting, goodbye, thanks, name, capabilities, joke, time, date, weather, motivation, study tips, how-are-you, creator, music, plus seven AI-topic intents: machine learning, deep learning, neural networks, speech recognition, NLP, artificial intelligence, and a comparison intent for "difference between X and Y" questions), **233 example utterances** (8–14 per intent), each intent with 1–3 canned responses. Time/date intents fill in the live value at reply time.
Preprocessing: lowercase, regex word tokenization, unigrams + bigrams, vocabulary of **694 features**, binary bag-of-words vectors.

## 3. Methodology
1. **Speech recognition:** browser Web Speech API (`SpeechRecognition`, en-US) streams microphone audio to the browser's built-in recognizer and returns a transcript; interim results are shown live. This is a pretrained deep-learning ASR system, so no server-side audio handling is needed.
2. **Intent classification:** transcript → `POST /chat` (Flask) → bag-of-words vector → PyTorch MLP → softmax over 21 intents.
3. **Response:** a random response for the predicted intent is returned. If the top softmax probability is below 0.55 the bot says it did not understand, rather than guessing.
4. **Voice output:** the reply is spoken with the browser's `speechSynthesis`.

## 4. Model architecture
`Linear(694→128) → ReLU → Dropout(0.4) → Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→21)`
Loss: cross-entropy with label smoothing 0.1 (reduces overconfident predictions). Optimizer: Adam, lr 3e-3, 300 full-batch epochs. Dropout is high because the dataset is small.

## 5. Results
Stratified 75/25 split (179 train / 54 test, seed 0):

| Set | Accuracy |
|---|---|
| Train | 100% |
| Held-out test | **75.9%** |

The shipped model is retrained on all 233 samples. The six AI-topic intents overlap heavily in vocabulary, which is why held-out accuracy is lower than a version with a single AI intent (81% on 15 intents) even though each topic now gets its own correct answer. Spot checks: "What is machine learning", "What is speech and language processing", "how does speech recognition work" and "explain neural networks" each map to their own intent (92–94% confidence); out-of-domain input ("what's the capital of France", "I like pizza") falls below the 0.55 threshold and gets the fallback reply. Comparison questions ("difference between machine learning and deep learning") initially failed in manual testing and were fixed by adding the comparison intent; those phrasings are now in the training data, so they no longer count as a fair test. The UI shows the top-3 intent probabilities for every utterance.

## 6. Limitations / future work
- Small hand-written dataset and a bag-of-words model: it fails on paraphrases that share no words with the training set. More data, or a pretrained embedding/transformer encoder, would improve it.
- ASR quality depends on the browser (Chrome/Edge/Safari) and sends audio to the browser vendor's service. Firefox does not support it; typed input is the fallback.
- No dialogue memory; each message is classified independently. The weather intent has no live data.

## 7. Deployment
Hosted free on a Hugging Face **static Space** (HTTPS, required by the microphone API): https://huggingface.co/spaces/adiag1205/voicebot. Free tiers do not offer a Python server, so `export_web.py` exports the trained PyTorch weights to `model.json` and the same forward pass runs in the browser in JavaScript (parity-checked against PyTorch on sample inputs). The Flask + Docker server version (`app.py`, `Dockerfile`, `render.yaml`) is kept in the repository for server-side hosting. First load of the site can take up to 1 minute.
