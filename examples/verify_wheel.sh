#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python -m venv "$TMP/venv"
"$TMP/venv/bin/pip" --disable-pip-version-check install --no-deps "$ROOT"
cd "$TMP"
"$TMP/venv/bin/python" - <<'PY'
import importlib,json
from importlib import resources
from pathlib import Path
from embedkit.model import train
from embedkit.inspector import inspect
from embedkit.capture import capture
from embedkit.evaluate import coherence,separation
for name in ('embedkit','embedkit.model','embedkit.evaluate','embedkit.report','embedkit.tokenize','embedkit.cli','embedkit.inspector','embedkit.capture'):
    importlib.import_module(name)
data=resources.files('embedkit').joinpath('data')
f=json.loads(data.joinpath('original.json').read_text())
emb=train(f['sentences'],**f['params'])
for w,v in f['vectors'].items():
    assert all(abs(a-b)<1e-9 for a,b in zip(v,emb.vector(w)))
assert coherence(emb,f['clusters'])==f['coherence'] and separation(emb,f['clusters'])==f['separation']
ws=[w for i in range(6) for w in json.loads(data.joinpath(f'witness-{i:02d}.json').read_text())]
assert len(ws)==19
for w in ws:
    assert inspect(w['value'])==w['report']
assert len(capture(f['sentences'],**f['params']))==12
Path('corpus.txt').write_text(f['corpus']);Path('eval.json').write_text(json.dumps({'clusters':f['clusters']}))
Path('input.json').write_text(json.dumps(ws[12]['value']))
print('Installed outside checkout: 8 modules, all 32 original vectors/300 sentences, 19 complete reports, 72 captured original events')
PY
"$TMP/venv/bin/embedkit" nearest --corpus corpus.txt --word gradient --dim 16 --epochs 30 --seed 7
"$TMP/venv/bin/embedkit" evaluate --corpus corpus.txt --eval eval.json --dim 16 --epochs 30 --seed 7 --min-coherence .9 --min-margin .4
"$TMP/venv/bin/embedkit" inspect --input input.json > report.json
