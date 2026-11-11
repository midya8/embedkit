"""Exact finite-retry laws under independent uniform-grid proposals, not PRNG law."""
from fractions import Fraction as F
from collections import Counter
import inspect,json,math,random,re
from .model import _negative_sampler


def ratio(x):return str(x.numerator)+'/'+str(x.denominator)
def record(x):return {'exact':ratio(x),'decimal':float(x)}
def integer(x,lo,hi):
    if type(x)is not int or not lo<=x<=hi:raise ValueError('integer outside bounds')
    return x

def validate(v):
    if type(v)is not dict or set(v)!={'words','cumulative','excluded','draws','attempts','grid_bits'}:raise ValueError('expected words,cumulative,excluded,draws,attempts,grid_bits only')
    w,c,e=v['words'],v['cumulative'],v['excluded']
    if type(w)is not list or not 2<=len(w)<=32 or any(type(x)is not str or not re.fullmatch('[a-z][a-z0-9_-]{0,23}',x)for x in w)or w!=sorted(set(w)):raise ValueError('2..32 sorted unique ASCII word labels required')
    if type(c)is not list or len(c)!=len(w)or any(type(x)not in(int,float)or not math.isfinite(x)or not 0<=x<=1+1e-12 for x in c)or c!=sorted(c):raise ValueError('nondecreasing finite cumulative thresholds required')
    if type(e)is not list or any(type(i)is not int or not 0<=i<len(w)for i in e)or e!=sorted(set(e))or len(e)==len(w):raise ValueError('sorted unique proper excluded indices required')
    k=integer(v['draws'],0,len(w)-len(e));integer(v['attempts'],1,20);integer(v['grid_bits'],1,53)
    if len(w)>8 and k>1:raise ValueError('complete multi-draw audit limited to8 words; up to32 only for zero/one draw')
    return v


def strict_json(text):
    if type(text)is not str or len(text.encode())>50000:raise ValueError('input exceeds50KB')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate key')
            d[k]=v
        return d
    try:v=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _:None);return validate(v)
    except (json.JSONDecodeError,RecursionError,OverflowError)as e:raise ValueError('invalid JSON')from e


def from_counts(counts,positive,draws=1,attempts=20,grid_bits=53):
    """Read thresholds actually made by the unchanged original sampler closure."""
    if type(counts)is not dict or not 2<=len(counts)<=32 or any(type(w)is not str or not re.fullmatch('[a-z][a-z0-9_-]{0,23}',w)for w in counts)or any(type(n)is not int or not 1<=n<=1000000 for n in counts.values()):raise ValueError('bounded positive integer counts required')
    draw=_negative_sampler(Counter(counts),random.Random(0));closure=inspect.getclosurevars(draw).nonlocals
    words=list(closure['words']);v={'words':words,'cumulative':list(closure['cumulative']),'excluded':[words.index(positive)],'draws':draws,'attempts':attempts,'grid_bits':grid_bits};validate(v);return v


def bins(v):
    d=1<<v['grid_bits'];ends=[0]+[max(0,min(d,math.ceil(F(float(x))*d)))for x in v['cumulative'][:-1]]+[d]
    return [b-a for a,b in zip(ends,ends[1:])],ends


def transition(masses,excluded,attempts):
    rejected=sum((masses[i]for i in excluded),F(0));eligible=[i for i in range(len(masses))if i not in excluded];first=eligible[0]
    factor=sum((rejected**r for r in range(attempts)),F(0));fallback=rejected**attempts
    actual={i:masses[i]*factor+(fallback if i==first else 0)for i in eligible}
    ideal=None if rejected==1 else{i:masses[i]/(1-rejected)for i in eligible}
    assert sum(actual.values())==1
    tv=None if ideal is None else sum((abs(actual[i]-ideal[i])for i in eligible),F(0))/2
    return actual,ideal,{'rejected_mass':record(rejected),'fallback_mass':record(fallback),'expected_proposals':record(factor),'fallback_index':first,'total_variation_to_conditional':None if tv is None else record(tv)}


def audit(v):
    validate(v);counts,ends=bins(v);d=1<<v['grid_bits'];p=[F(n,d)for n in counts];base=set(v['excluded']);k=v['draws'];actual={0:F(1)};ideal={0:F(1)};states=[];layers=[];fallback_total=F(0);proposals=F(0);ideal_stalled=F(0)
    for step in range(k+1):
        layers.append({'draw':step,'actual_mass':record(sum(actual.values(),F(0))),'conditional_active_mass':record(sum(ideal.values(),F(0))),'conditional_stalled_mass':record(ideal_stalled)})
        nxt={};nxti={}
        for mask in sorted(set(actual)|set(ideal)):
            chosen=[i for i in range(len(p))if mask>>i&1];excluded=base|set(chosen);am=actual.get(mask,F(0));im=ideal.get(mask,F(0));entry={'draw':step,'mask':mask,'chosen':chosen,'actual_mass':record(am),'conditional_mass':record(im),'transition':None}
            if step<k:
                a,b,meta=transition(p,excluded,v['attempts']);entry['transition']={**meta,'outcomes':[{'index':i,'actual':record(a[i]),'conditional':None if b is None else record(b[i])}for i in a]}
                fallback_total+=am*F(meta['fallback_mass']['exact']);proposals+=am*F(meta['expected_proposals']['exact'])
                for i,q in a.items():
                    m=mask|1<<i
                    if q and am:nxt[m]=nxt.get(m,F(0))+am*q
                if b is None:ideal_stalled+=im
                else:
                    for i,q in b.items():
                        m=mask|1<<i
                        if q and im:nxti[m]=nxti.get(m,F(0))+im*q
            states.append(entry)
        if step<k:actual,ideal=nxt,nxti
    inclusion=[]
    for i in range(len(p)):
        a=sum((q for m,q in actual.items()if m>>i&1),F(0));b=sum((q for m,q in ideal.items()if m>>i&1),F(0));inclusion.append({'index':i,'actual':record(a),'conditional':None if ideal_stalled else record(b),'difference':None if ideal_stalled else record(a-b)})
    assert sum((F(x['actual']['exact'])for x in inclusion),F(0))==k
    final_tv=None if ideal_stalled else sum((abs(actual.get(m,F(0))-ideal.get(m,F(0)))for m in set(actual)|set(ideal)),F(0))/2
    return {'schema':'embedkit-sampler/1','input':v,'grid_size':str(d),'intervals':[{'index':i,'start':str(ends[i]),'stop':str(ends[i+1]),'grid_count':str(counts[i]),'mass':record(p[i])}for i in range(len(p))],
            'states':states,'layers':layers,'inclusion':inclusion,'expected_fallback_events':record(fallback_total),'expected_proposals':record(proposals),'final_subset_tv':None if final_tv is None else record(final_tv),
            'scope':'Exact under independent uniform grid proposals, not the deterministic PRNG joint law or fixed-seed probability. Subset law and inclusion weights, not expected sequential SGD updates, training quality or a new sampling method.'}
