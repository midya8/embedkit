"""Bounded sampled logistic objective and original sequential update audit.

This is a new inspector, not a replacement trainer or a noise-distribution
estimator. The original trainer's clamp and event ordering are replayed exactly.
"""
from __future__ import annotations
import copy
import json
import math


def total(values):
    # Python 3.12 changed sum(float); use sequential binary64 in both languages.
    out = 0.0
    for v in values:
        out += v
    return out


def number(x, limit=100):
    if type(x) not in (int, float) or not math.isfinite(x) or abs(x) > limit:
        raise ValueError('finite numeric value outside bounds')
    return x


def validate(v):
    if not isinstance(v, dict) or set(v) != {'center', 'outputs', 'events', 'lr'}:
        raise ValueError('expected center, outputs, events, lr only')
    c, o, e = v['center'], v['outputs'], v['events']
    if not isinstance(c, list) or not 1 <= len(c) <= 16:
        raise ValueError('center dimension must be 1..16')
    for x in c:
        number(x)
    if not isinstance(o, list) or not 1 <= len(o) <= 12:
        raise ValueError('output table must have 1..12 rows')
    for row in o:
        if not isinstance(row, list) or len(row) != len(c):
            raise ValueError('rectangular outputs with center dimension required')
        for x in row:
            number(x)
    if not isinstance(e, list) or not 1 <= len(e) <= 33:
        raise ValueError('events must have 1..33 rows')
    for event in e:
        if not isinstance(event, dict) or set(event) != {'row', 'label'}:
            raise ValueError('event requires row and label only')
        if type(event['row']) is not int or not 0 <= event['row'] < len(o):
            raise ValueError('invalid output row index')
        if type(event['label']) is not int or event['label'] not in (0, 1):
            raise ValueError('label must be integer 0 or 1')
    number(v['lr'], 2)
    if v['lr'] <= 0:
        raise ValueError('learning rate must be > 0 and <= 2')
    return v


def strict_json(text):
    if len(text.encode('utf-8')) > 50000:
        raise ValueError('input exceeds 50KB')
    depth = 0
    string = escape = False
    for ch in text:
        if string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                string = False
        elif ch == '"':
            string = True
        elif ch in '[{':
            depth += 1
            if depth > 8:
                raise ValueError('nesting exceeds 8')
        elif ch in ']}':
            depth -= 1
    def pairs(items):
        out = {}
        for k, x in items:
            if k in out:
                raise ValueError('duplicate key')
            out[k] = x
        return out
    def bad(x):
        raise ValueError('nonfinite JSON number')
    try:
        return validate(json.loads(text, object_pairs_hook=pairs, parse_constant=bad))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid JSON') from exc


def sigmoid(s):
    if s >= 0:
        return 1 / (1 + math.exp(-s))
    t = math.exp(s)
    return t / (1 + t)


def term(c, row, label):
    dot = total(a*b for a,b in zip(c,row))
    z = -dot if label else dot
    loss = max(0, z) + math.log1p(math.exp(-abs(z)))
    # Avoid subtracting one from a rounded-to-one positive sigmoid.
    factor = -sigmoid(-dot) if label else sigmoid(dot)
    return dot, loss, factor


def objective(c, outputs, events):
    return total(term(c, outputs[e['row']], e['label'])[1] for e in events)


def inspect(v):
    validate(v)
    c, o, events, lr = v['center'], v['outputs'], v['events'], v['lr']
    gc = [0.0]*len(c)
    go = [[0.0]*len(c) for _ in o]
    frozen = []
    for event in events:
        j, label = event['row'], event['label']
        dot, loss, factor = term(c,o[j],label)
        dc = [factor*x for x in o[j]]
        do = [factor*x for x in c]
        for k in range(len(c)):
            gc[k] += dc[k]
            go[j][k] += do[k]
        frozen.append(dict(event, dot=dot, loss=loss, factor=factor,
                           center_gradient=dc, output_gradient=do))
    sc, so = c[:], copy.deepcopy(o)
    trajectory = []
    for event in events:
        j, label = event['row'], event['label']
        before_c, before_o = sc[:], so[j][:]
        dot, loss, true_factor = term(sc,so[j],label)
        clipped = max(-15.0,min(15.0,dot))
        # Exact original operation order: grad includes lr before vector multiply.
        factor = 1/(1+math.exp(-clipped))-label
        grad = factor*lr
        for k in range(len(c)):
            old = sc[k]
            sc[k] -= grad*so[j][k]
            so[j][k] -= grad*old
        trajectory.append(dict(event, dot=dot, clipped_dot=clipped,
            was_clipped=dot!=clipped, factor=factor, true_factor=true_factor,
            event_loss_before=loss,event_loss_after=term(sc,so[j],label)[1],
            center_before=before_c, output_before=before_o,
            center_after=sc[:], output_after=so[j][:]))
    bc = [x-lr*g for x,g in zip(c,gc)]
    bo = [[x-lr*g for x,g in zip(row,gr)] for row,gr in zip(o,go)]
    # Central differences independently evaluate the unclipped frozen loss.
    checks = []
    for name, table, gradients in [('center',[c],[gc]),('outputs',o,go)]:
        for j,row in enumerate(table):
            for k,x in enumerate(row):
                h = 1e-5*max(1,abs(x))
                plus,minus = copy.deepcopy(v),copy.deepcopy(v)
                if name=='center':
                    plus[name][k] += h; minus[name][k] -= h
                else:
                    plus[name][j][k] += h; minus[name][j][k] -= h
                numeric=(objective(plus['center'],plus['outputs'],events)-objective(minus['center'],minus['outputs'],events))/(2*h)
                analytic=gradients[j][k]
                checks.append({'table':name,'row':j,'coordinate':k,'analytic':analytic,'numeric':numeric,
                               'absolute_error':abs(numeric-analytic),
                               'normalized_error':abs(numeric-analytic)/max(1,abs(numeric),abs(analytic))})
    loss_before=objective(c,o,events)
    return {'input':copy.deepcopy(v),'frozen':frozen,'center_gradient':gc,'output_gradients':go,
        'loss_before':loss_before,
        'sequential':{'center':sc,'outputs':so,'loss_after':objective(sc,so,events),'trajectory':trajectory},
        'simultaneous':{'center':bc,'outputs':bo,'loss_after':objective(bc,bo,events)},
        'update_l2_difference':math.sqrt(total((a-b)**2 for a,b in zip(sc,bc))+total((a-b)**2 for ar,br in zip(so,bo) for a,b in zip(ar,br))),
        'gradient_checks':checks,'max_gradient_error':max(x['normalized_error'] for x in checks),
        'scope':'Fixed ordered sampled events, not expected noise loss or calibrated word probabilities. Sequential replay uses original +/-15 logit clamp; simultaneous uses the unclipped frozen objective. No descent or quality guarantee.'}
