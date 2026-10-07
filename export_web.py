# Builds space/: a static, serverless copy of the app. Same trained weights, inference in JS.
import json, os, torch
ck = torch.load("model.pt"); s = ck["state"]
W = lambda k: [[round(x, 5) for x in r] for r in s[k].tolist()]
B = lambda k: [round(x, 5) for x in s[k].tolist()]
resp = {i["tag"]: i["responses"] for i in json.load(open("data/intents.json"))["intents"]}
os.makedirs("space", exist_ok=True)
json.dump({"vocab": ck["vocab"], "tags": ck["tags"], "responses": resp,
           "layers": [[W(f"net.{i}.weight"), B(f"net.{i}.bias")] for i in (0, 3, 6)]},
          open("space/model.json", "w"), separators=(",", ":"))

html = open("static/index.html").read()
start = html.index("async function ask(text){"); end = html.index("const SR=")
local = '''let M;const modelP=fetch("model.json").then(r=>r.json()).then(m=>M=m);
function tok(t){const w=t.toLowerCase().match(/[a-z']+/g)||[];return w.concat(w.slice(1).map((x,i)=>w[i]+"_"+x))}
function classify(text){
  let x=new Array(Object.keys(M.vocab).length).fill(0);for(const t of tok(text))if(t in M.vocab)x[M.vocab[t]]=1;
  M.layers.forEach(([W,b],k)=>{x=W.map((r,i)=>r.reduce((s,v,j)=>s+v*x[j],b[i]));if(k<2)x=x.map(v=>Math.max(0,v))});
  const mx=Math.max(...x),e=x.map(v=>Math.exp(v-mx)),z=e.reduce((a,b)=>a+b),p=e.map(v=>v/z),i=p.indexOf(Math.max(...p));
  if(p[i]<0.55||!text.trim())return{intent:"unknown",confidence:p[i],response:"Sorry, I didn't understand that. Try asking for a joke, the time or my name."};
  const tag=M.tags[i],rs=M.responses[tag],n=new Date();
  const r=rs[Math.floor(Math.random()*rs.length)].replace("__TIME__","It's "+n.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})+".").replace("__DATE__","Today is "+n.toLocaleDateString([],{weekday:"long",day:"numeric",month:"long",year:"numeric"})+".");
  return{intent:tag,confidence:p[i],response:r}}
async function ask(text){
  if(!text.trim())return;add("u",text);await modelP;const r=classify(text);
  add("b",r.response,`intent: ${r.intent} · confidence ${(r.confidence*100).toFixed(0)}%`);
  speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(r.response));
}
'''
open("space/index.html", "w").write(html[:start] + local + html[end:])
open("space/README.md", "w").write("---\ntitle: VoiceBot\nemoji: 🎤\nsdk: static\n---\nVoice chatbot: browser speech recognition + PyTorch-trained intent network running in-browser. Source: https://github.com/karanjain10/voicebot\n")
