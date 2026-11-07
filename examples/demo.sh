#!/usr/bin/env bash
# Fully offline EmbedKit demo: train SGNS embeddings on a checked-in
# three-topic corpus, inspect neighbors, then exercise both gate sides.
set -u
cd "$(dirname "$0")/.."

OUT="$(mktemp -d /tmp/embedkit-demo-XXXX)"
trap 'echo "demo artifacts: $OUT"' EXIT

echo "== nearest neighbors (expect same-topic words) =="
python3 -m embedkit nearest --corpus examples/data/corpus.txt \
  --word gradient --dim 16 --epochs 30 --seed 7 || { echo "FAIL nearest"; exit 1; }

echo "== evaluate (expect pass) =="
python3 -m embedkit evaluate --corpus examples/data/corpus.txt \
  --eval examples/data/eval.json --dim 16 --epochs 30 --seed 7 \
  --json "$OUT/report.json" --markdown "$OUT/report.md" \
  --min-coherence 0.9 --min-margin 0.4 || { echo "FAIL quality gate that should pass"; exit 1; }

python3 - "$OUT/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
coh, sep = report["coherence"], report["separation"]
assert coh["words"] == 24 and coh["mean_coherence"] >= 0.9
assert sep["margin"] >= 0.4 and sep["mean_intra"] > sep["mean_inter"]
assert all(len(d["nearest"]) == 3 for d in coh["per_word"].values())
print("assertions ok: coherence %s, margin %s" % (coh["mean_coherence"], sep["margin"]))
PY

echo "== gate (expect fail) =="
if python3 -m embedkit evaluate --corpus examples/data/corpus.txt \
  --eval examples/data/eval.json --dim 16 --epochs 30 --seed 7 \
  --min-coherence 1.01 >/dev/null 2>&1; then
  echo "FAIL impossible gate passed"; exit 1
fi
echo "impossible coherence gate rejected as expected"

echo "DEMO OK"
