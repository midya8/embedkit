"""Regenerate full original fixture and complete event/gradient witnesses."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embedkit.model import train
from embedkit.tokenize import tokenize_sentences
from embedkit.evaluate import coherence,separation
from embedkit.capture import capture
from embedkit.inspector import inspect
root=Path(__file__).resolve().parents[1]
def save(path,v):
    path.write_text(json.dumps(v,separators=(',',':'),sort_keys=True)+'\n')
text=(root/'examples/data/corpus.txt').read_text()
clusters=json.loads((root/'examples/data/eval.json').read_text())['clusters']
sentences=tokenize_sentences(text)
params=dict(dim=16,window=2,negatives=5,epochs=30,lr=.05,seed=7,min_count=1)
emb=train(sentences,**params)
fixture={'corpus':text,'sentences':sentences,'clusters':clusters,'params':params,
 'vectors':{w:list(emb.vector(w)) for w in sorted(emb.vocab)},
 'coherence':coherence(emb,clusters),'separation':separation(emb,clusters),
 'all_neighbors':{w:emb.nearest(w,len(emb.vocab)-1) for w in sorted(emb.vocab)},
 'capture_runtime':'CPython 3.10; original source unchanged; full fixture numeric tolerance 1e-9 across supported runtimes'}
save(root/'embedkit/data/original.json',fixture)
witnesses=capture(sentences,**params)
controls=[
 ('repeated-negative',{'center':[.6,-.4,.2],'outputs':[[.3,.7,-.2],[-.8,.1,.5],[.2,-.6,.4]],'events':[{'row':0,'label':1},{'row':1,'label':0},{'row':1,'label':0},{'row':2,'label':0}],'lr':.2}),
 ('contradictory-row',{'center':[.8,-.2],'outputs':[[.4,.6],[.3,-.1]],'events':[{'row':0,'label':1},{'row':0,'label':0},{'row':1,'label':0}],'lr':.3}),
 ('zero-output-init',{'center':[.2,-.1,.3],'outputs':[[0.,0.,0.],[0.,0.,0.]],'events':[{'row':0,'label':1},{'row':1,'label':0}],'lr':.1}),
 ('saturated-clamp',{'center':[5.,-4.],'outputs':[[5.,-4.],[-5.,4.]],'events':[{'row':0,'label':1},{'row':1,'label':0}],'lr':1.}),
 ('overshoot',{'center':[2.,2.],'outputs':[[2.,2.],[2.,2.]],'events':[{'row':0,'label':0},{'row':1,'label':0}],'lr':2.}),
 ('one-row',{'center':[0.],'outputs':[[0.]],'events':[{'row':0,'label':1}],'lr':.1}),
 ('max-shape',{'center':[(-1)**j*(j+1)/16 for j in range(16)],'outputs':[[((i*3+j*7)%19-9)/10 for j in range(16)] for i in range(12)],'events':[{'row':i%12,'label':int(i==0)} for i in range(33)],'lr':.03}),
]
for name,v in controls:
    witnesses.append({'name':name,'kind':'constructed-control','center_word':'input role','words':[f'output-{j}' for j in range(len(v['outputs']))],'value':v,'report':inspect(v)})
for old in (root/'embedkit/data').glob('witness-*.json'):
    old.unlink()
chunks=[witnesses[i:i+3] for i in range(0,12,3)]+[witnesses[12:-1],witnesses[-1:]]
for i,chunk in enumerate(chunks):
    (root/f'embedkit/data/witness-{i:02d}.json').write_text(json.dumps(chunk,separators=(',',':'),sort_keys=True)+'\n')
# The browser ships inputs, original vectors, and source evidence, not Python reports.
manifest=[{k:v for k,v in w.items() if k not in ('report','original_returns')} for w in witnesses]
save(root/'embedkit/data/inputs.json',manifest)
(root/'docs/data.js').write_text('window.EMBED_DATA='+json.dumps({'original':fixture,'fixtures':manifest},separators=(',',':'))+';\n')
paths=['embedkit/model.py','embedkit/evaluate.py','embedkit/report.py','embedkit/tokenize.py','examples/demo.sh','examples/data/corpus.txt','examples/data/eval.json','examples/data/gen_corpus.py']+[str(p.relative_to(root)) for p in sorted((root/'tests').glob('test_*.py')) if p.name!='test_inspector.py']
save(root/'embedkit/data/original-hashes.json',{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
print('Original',len(emb.vocab),'words',len(sentences),'sentences',fixture['coherence']['mean_coherence'],fixture['separation']['margin'])
print(len(witnesses),'complete reports; original events',sum(len(w['value']['events']) for w in witnesses if w['kind']=='captured-original'),'numeric checks',sum(len(w['report']['gradient_checks']) for w in witnesses))
