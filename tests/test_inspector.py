import copy,hashlib,json,math,sys,unittest
from pathlib import Path
from embedkit.inspector import inspect,strict_json,validate,term
from embedkit.capture import capture
from embedkit.model import train
from embedkit.tokenize import tokenize_sentences
from embedkit.evaluate import coherence,separation
ROOT=Path(__file__).resolve().parents[1]
def witnesses():
    return [w for p in sorted((ROOT/'embedkit/data').glob('witness-*.json')) for w in json.loads(p.read_text())]
class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws=witnesses();cls.by={w['name']:w for w in cls.ws}
    def test_full_python_reports(self):
        for w in self.ws:
            self.assertEqual(inspect(w['value']),w['report'],w['name'])
    def test_all_finite_difference_coordinates(self):
        self.assertEqual(sum(len(w['report']['gradient_checks']) for w in self.ws),1593)
        self.assertLess(max(w['report']['max_gradient_error'] for w in self.ws),1e-8)
    def test_repeated_rows_accumulate_not_deduplicate(self):
        r=self.by['repeated-negative']['report'];es=[x for x in r['frozen'] if x['row']==1]
        self.assertEqual(len(es),2)
        self.assertEqual(r['output_gradients'][1],[2*x for x in es[0]['output_gradient']])
    def test_conflicting_labels_same_output_role(self):
        r=self.by['contradictory-row']['report']
        self.assertEqual(r['frozen'][0]['row'],r['frozen'][1]['row'])
        self.assertNotEqual(r['sequential']['center'],r['simultaneous']['center'])
    def test_clamp_not_true_gradient(self):
        r=self.by['saturated-clamp']['report']
        self.assertTrue(all(e['was_clipped'] for e in r['sequential']['trajectory']))
        self.assertGreater(abs(r['sequential']['trajectory'][0]['factor']),abs(r['frozen'][0]['factor'])*1e6)
    def test_large_rate_does_not_guarantee_descent(self):
        r=self.by['overshoot']['report']
        self.assertGreater(r['simultaneous']['loss_after'],r['loss_before'])
    def test_input_not_mutated(self):
        v=copy.deepcopy(self.ws[0]['value']);before=copy.deepcopy(v);inspect(v);self.assertEqual(v,before)
    def test_stable_extreme_terms(self):
        self.assertTrue(math.isfinite(term([100],[100],0)[1]))
        self.assertEqual(term([100],[100],0)[1],10000)
        self.assertEqual(term([-100],[100],1)[1],10000)
        self.assertGreater(term([5],[5],1)[1],0)
    def test_strict_json(self):
        valid=json.dumps(self.ws[0]['value']);self.assertEqual(strict_json(valid),self.ws[0]['value'])
        for s in ['{"center":[],"center":[]}',valid+'x','{"x":NaN}', '['*10+'0'+']'*10,valid+' '*50001]:
            with self.assertRaises(ValueError):strict_json(s)
    def test_schema_bounds(self):
        v=self.ws[0]['value']
        for k,x in [('center',[]),('center',[True]),('center',[math.inf]),('outputs',[[0]]),('events',[{'row':0,'label':True}]),('events',[{'row':-1,'label':0}]),('lr',0),('lr',3),('lr','0.1')]:
            bad=copy.deepcopy(v);bad[k]=x
            with self.assertRaises(ValueError,msg=str((k,x))):validate(bad)
        bad=dict(v,extra=1)
        with self.assertRaises(ValueError):validate(bad)
    def test_original_files_untouched(self):
        for p,h in json.loads((ROOT/'embedkit/data/original-hashes.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),h,p)
    def test_retrace_real_original_calls(self):
        f=json.loads((ROOT/'embedkit/data/original.json').read_text());old=sys.gettrace()
        fresh=capture(f['sentences'],**f['params']);self.assertIs(sys.gettrace(),old)
        self.assertEqual(len(fresh),12)
        for a,b in zip(fresh,self.ws[:12]):
            self.assertEqual(a['words'],b['words']);self.assertEqual(a['center_word'],b['center_word']);self.assertEqual(a['value']['events'],b['value']['events'])
            for av,bv in zip(a['report']['sequential']['center'],b['report']['sequential']['center']):self.assertAlmostEqual(av,bv,places=12)
    def test_complete_original_fixture_retrains(self):
        f=json.loads((ROOT/'embedkit/data/original.json').read_text());self.assertEqual(tokenize_sentences(f['corpus']),f['sentences'])
        emb=train(f['sentences'],**f['params']);self.assertEqual(emb.vocab,frozenset(f['vectors']))
        for w,vec in f['vectors'].items():
            for a,b in zip(emb.vector(w),vec):self.assertAlmostEqual(a,b,places=9)
        self.assertEqual(coherence(emb,f['clusters']),f['coherence']);self.assertEqual(separation(emb,f['clusters']),f['separation'])
