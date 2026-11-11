"""Deterministic sampler inputs plus separate full native reports; no old fixture edits."""
import json,random,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embedkit.sampler import from_counts,audit
from embedkit.tokenize import tokenize_sentences
P=Path(__file__).resolve().parents[1];fixtures=[]
def add(name,v,kind='constructed-control',counts=None):
    fixtures.append({'name':name,'kind':kind,'counts':counts,'input':v,'report':audit(v)})
def counts(name,c,positive,k=1):add(name,from_counts(c,positive,k),counts=c)
counts('dominant-positive',{'a':1,'b':1,'z':1000000},'z')
counts('balanced-three-draws',{'a':1,'b':1,'c':1,'d':1},'d',3)
counts('skewed-two-draws',{'a':1,'b':2,'c':3,'z':10000},'z',2)
counts('center-still-eligible',{'center':4,'context':12,'other':1},'context',2)
counts('eight-word-two-draws',{chr(97+i):i+1 for i in range(8)},'h',2)
add('zero-eligible-mass',{'words':['a','b','c'],'cumulative':[0,0,1],'excluded':[2],'draws':2,'attempts':20,'grid_bits':53})
add('boundary-and-unused-tail',{'words':['a','b','c','d'],'cumulative':[.125,.125,.5000000000000001,.9],'excluded':[0],'draws':2,'attempts':3,'grid_bits':3})
add('no-draw-control',{'words':['a','b'],'cumulative':[.5,1],'excluded':[],'draws':0,'attempts':20,'grid_bits':53})
original=json.loads((P/'embedkit/data/original.json').read_text());cc=Counter(w for sentence in original['sentences']for w in sentence);v=from_counts(dict(sorted(cc.items())),'gradient',1);add('original-corpus-one-draw',v,'original-corpus-counts',dict(sorted(cc.items())))
rng=random.Random(791)
for j in range(24):
    n=rng.randint(2,5);c={chr(97+i):rng.randint(1,1000000)for i in range(n)};v=from_counts(c,chr(96+n),rng.randint(0,min(n-1,3)));v['attempts']=rng.randint(1,20);add('seeded-law-'+str(j),v,'seeded-check',c)
for i in range(0,len(fixtures),3):(P/f'embedkit/data/sampler-witness-{i//3:02d}.json').write_text(json.dumps(fixtures[i:i+3],sort_keys=True,separators=(',',':'))+'\n')
public=[{k:v for k,v in w.items()if k!='report'}for w in fixtures]
(P/'embedkit/data/sampler-inputs.json').write_text(json.dumps(public,sort_keys=True,separators=(',',':'))+'\n')
(P/'docs/sampler-data.js').write_text('window.SAMPLER_INPUTS='+json.dumps(public,separators=(',',':'))+';\n')
print(len(fixtures),'full sampler reports;',sum(len(w['report']['states'])for w in fixtures),'DP states; original counts',sum(cc.values()),'tokens',len(cc),'words')
