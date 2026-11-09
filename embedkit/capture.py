"""Read-only Python trace capture of unchanged train()'s ordered step calls.

No monkeypatch, modified trainer or replacement RNG. A private exception stops
this bounded prefix before pair 13. Only direct calls to the nested step are
recorded. The full fixture is trained separately without tracing.
"""
import copy
import sys
from .model import train
from .inspector import inspect


def capture(sentences, *, pairs=12, **params):
    if type(pairs) is not int or not 1 <= pairs <= 16:
        raise ValueError('pairs must be 1..16')
    code = next(c for c in train.__code__.co_consts if hasattr(c,'co_name') and c.co_name=='step')
    groups=[]
    class StopCapture(Exception):
        pass
    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        s=frame.f_locals
        if event=='call':
            parent=frame.f_back.f_locals
            if s['label']==1:
                if len(groups)==pairs:
                    raise StopCapture
                groups.append({'center_word':s['center'], 'center':parent['center_w'][s['center']][:],
                    'all_outputs':copy.deepcopy(parent['context_w']), 'events':[], 'returns':[]})
            groups[-1]['events'].append({'word':s['context'],'label':s['label']})
        elif event=='return':
            groups[-1]['returns'].append({'center':s['cv'][:],'output':s['xv'][:]})
        return trace
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        train(sentences,**params)
    except StopCapture:
        pass
    finally:
        sys.settrace(old)
    fixtures=[]
    for n,g in enumerate(groups):
        words=list(dict.fromkeys(e['word'] for e in g['events']))
        value={'center':g['center'],'outputs':[g['all_outputs'][w] for w in words],
               'events':[{'row':words.index(e['word']),'label':e['label']} for e in g['events']],
               'lr':params.get('lr',.05)}
        report=inspect(value)
        assert len(report['sequential']['trajectory'])==len(g['returns'])
        for s,r in zip(report['sequential']['trajectory'],g['returns']):
            assert all(abs(a-b) <= 1e-13*max(1,abs(a),abs(b)) for key, other in [('center_after','center'),('output_after','output')] for a,b in zip(s[key],r[other]))
        fixtures.append({'name':f'original-pair-{n+1:02d}','kind':'captured-original',
            'center_word':g['center_word'],'words':words,'value':value,'report':report,
            'original_returns':g['returns']})
    return fixtures
