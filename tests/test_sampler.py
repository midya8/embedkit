import copy,itertools,json,math,random,subprocess,sys,tempfile,unittest
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
from embedkit.sampler import audit,bins,from_counts,strict_json,transition,validate
from embedkit.model import _negative_sampler
ROOT=Path(__file__).resolve().parents[1]
class Scripted:
    def __init__(self,values):self.values=iter(values);self.calls=0
    def random(self):self.calls+=1;return next(self.values)
def original_bin(c,r):
    lo,hi=0,len(c)-1
    while lo<hi:
        mid=(lo+hi)//2
        if r<c[mid]:hi=mid
        else:lo=mid+1
    return lo
class SamplerTests(unittest.TestCase):
    def test_all_complete_packaged_reports(self):
        for f in sorted((ROOT/'embedkit/data').glob('sampler-witness-*.json')):
            for w in json.loads(f.read_text()):self.assertEqual(audit(w['input']),w['report'])
    def test_exhaustive_small_grid_draw_tree_oracle(self):
        rng=random.Random(791);cases=0;paths=0
        for _ in range(36):
            n=rng.randint(2,4);bits=rng.randint(1,3);D=1<<bits;A=rng.randint(1,3);cs=sorted([rng.randrange(D+1)/D for _ in range(n-1)])+[1.];v={'words':list('abcd')[:n],'cumulative':cs,'excluded':[n-1],'draws':rng.randint(1,n-1),'attempts':A,'grid_bits':bits}
            report=audit(v);dist={0:F(1)};proposal_total=F(0);fallback_total=F(0)
            # Enumerate complete fixed-length random tuples, not a geometric formula.
            for step in range(v['draws']):
                nxt=defaultdict(F)
                for mask,mass in dist.items():
                    ex={n-1}|{i for i in range(n)if mask>>i&1};hist=Counter();consumed=0;fallbacks=0
                    for draws in itertools.product(range(D),repeat=A):
                        paths+=1;answer=None
                        for t,m in enumerate(draws,1):
                            idx=original_bin(cs,m/D)
                            if idx not in ex:answer=idx;consumed+=t;break
                        if answer is None:answer=next(i for i in range(n)if i not in ex);consumed+=A;fallbacks+=1
                        hist[answer]+=1
                    for idx,num in hist.items():nxt[mask|1<<idx]+=mass*F(num,D**A)
                    proposal_total+=mass*F(consumed,D**A);fallback_total+=mass*F(fallbacks,D**A)
                dist=dict(nxt)
            for x in report['inclusion']:self.assertEqual(F(x['actual']['exact']),sum((q for m,q in dist.items()if m>>x['index']&1),F(0)))
            self.assertEqual(F(report['expected_proposals']['exact']),proposal_total);self.assertEqual(F(report['expected_fallback_events']['exact']),fallback_total)
            cases+=1
        print('exhaustive random-tuple oracle',cases,'cases',paths,'paths')
    def test_grid_boundaries_against_actual_original_binary_search(self):
        rng=random.Random(982);checked=0
        for _ in range(120):
            n=rng.randint(2,8);counts={chr(97+i):rng.randint(1,1000000)for i in range(n)};v=from_counts(counts,chr(96+n),1);nums,ends=bins(v);D=1<<53
            # Every threshold adjacent grid value, including exact and both neighbors.
            for end in ends:
                for m in {max(0,min(D-1,end+j))for j in(-1,0,1)}:
                    idx=original_bin(v['cumulative'],m/D);self.assertTrue(ends[idx]<=m<ends[idx+1]);checked+=1
                    fake=Scripted([m/D]);draw=_negative_sampler(Counter(counts),fake);self.assertEqual(draw(set()),v['words'][idx]);self.assertEqual(fake.calls,1)
        print('original sampler grid boundary calls',checked)
    def test_actual_original_retry_fallback_consumption_and_exclusions(self):
        counts=Counter({'a':1,'b':1,'z':1000000});v=from_counts(dict(counts),'z')
        for rejected in range(20):
            fake=Scripted([.9]*rejected+[0.]);draw=_negative_sampler(counts,fake);self.assertEqual(draw({'z'}),'a');self.assertEqual(fake.calls,rejected+1)
        fake=Scripted([.9]*20);self.assertEqual(_negative_sampler(counts,fake)({'z'}),'a');self.assertEqual(fake.calls,20)
        fake=Scripted([0.]*20);self.assertEqual(_negative_sampler(counts,fake)({'a','z'}),'b');self.assertEqual(fake.calls,20)
        # Center is eligible unless also the positive context; original function has no center argument.
        fake=Scripted([0.]);self.assertEqual(_negative_sampler(counts,fake)({'z'}),'a')
    def test_degenerate_zero_mass_and_final_threshold_unused(self):
        v={'words':['a','b','c'],'cumulative':[0.,0.,0.],'excluded':[2],'draws':2,'attempts':20,'grid_bits':53};r=audit(v)
        self.assertEqual(r['inclusion'][0]['actual']['exact'],'1/1');self.assertIsNone(r['final_subset_tv']);self.assertEqual(r['expected_proposals']['exact'],'40/1');self.assertEqual(r['expected_fallback_events']['exact'],'2/1')
        v['cumulative'][-1]=1.;self.assertEqual(bins(v)[0],[0,0,1<<53])
    def test_conservation_and_conditional_path_oracle(self):
        rng=random.Random(734);total=0
        for _ in range(32):
            n=rng.randint(2,6);v=from_counts({chr(97+i):rng.randint(1,1000)for i in range(n)},chr(96+n),rng.randint(0,n-1));v['attempts']=rng.randint(1,20);r=audit(v);nums,_=bins(v);p=[F(x,1<<53)for x in nums];expected=[F(0)]*n;k=v['draws']
            for path in itertools.permutations(range(n-1),k):
                q=F(1);ex={n-1}
                for i in path:q*=p[i]/(1-sum(p[j]for j in ex));ex.add(i)
                for i in path:expected[i]+=q
                total+=1
            for row,q in zip(r['inclusion'],expected):self.assertEqual(F(row['conditional']['exact']),q)
            self.assertEqual(sum(F(x['actual']['exact'])for x in r['inclusion']),k)
            for layer in r['layers']:self.assertEqual(layer['actual_mass']['exact'],'1/1')
        print('independent unlimited conditional ordered paths',total)
        full=audit(from_counts({chr(97+i):i+1 for i in range(8)},'h',7));self.assertEqual(full['final_subset_tv']['exact'],'0/1');self.assertEqual(sum(F(x['actual']['exact'])for x in full['inclusion']),7)
    def test_strict_input_limits(self):
        v=from_counts({'a':1,'b':2},'b')
        for change in [{'draws':True},{'attempts':21},{'grid_bits':54},{'words':['b','a']},{'cumulative':[1,.5]},{'excluded':[0,1]},{'excluded':[True]},{'extra':1},{'cumulative':[True,1]}]:
            with self.assertRaises((ValueError,TypeError)):validate({**v,**change})
        for text in ['{"a":1,"a":2}','NaN',' '.ljust(50001),'['*1000]:
            with self.assertRaises(ValueError):strict_json(text)
        with self.assertRaises(ValueError):from_counts({'a':0,'b':1},'b')
    def test_cli_replay(self):
        v=from_counts({'a':1,'b':2},'b')
        with tempfile.TemporaryDirectory()as td:
            f=Path(td)/'input.json';f.write_text(json.dumps(v));p=subprocess.run([sys.executable,'-m','embedkit.sampler_cli','--input',str(f)],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),audit(v));f.write_text('{}');p=subprocess.run([sys.executable,'-m','embedkit.sampler_cli','--input',str(f)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')
