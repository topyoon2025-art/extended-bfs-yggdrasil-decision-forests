#!/bin/bash
set -e
cd /home/ubuntu/projects/extended-bfs-yggdrasil-decision-forests
RUN="--input_mode=tfrecord --ds_path=/home/ubuntu/projects/dataset/HIGGS1 --label_col=target --num_trees=2 --tree_depth=10 --seed=1"

run_one () {
  local name="$1"; local bin="$2"
  echo "=== RUN $name ==="
  "$bin" $RUN 2> /tmp/run_${name}.err >/tmp/run_${name}.out || { echo "RUN FAILED: $name"; tail -5 /tmp/run_${name}.err; return 1; }
  cp -f per_depth_timing.csv /tmp/pd_${name}.csv
  echo "$name done"
}

run_one srdw_label /tmp/ap_srdw_label
run_one offsets_only /tmp/ap_offsets_only
run_one nid_srdw /tmp/ap_nid_srdw

echo
echo "=== build_hist (ms) per depth, averaged over trees — from stderr 'hist=' ==="
python3 -I - <<'PY'
import re, glob
variants = ["srdw_label","offsets_only","nid_srdw"]
data = {}   # variant -> depth -> [hist values]
nodes = {}  # variant -> depth -> nodes (last seen)
for v in variants:
    data[v] = {}; nodes[v] = {}
    for line in open(f"/tmp/run_{v}.err"):
        m = re.search(r'^D(\d+)\s+nodes=(\d+).*?hist=([0-9.eE+-]+)', line)
        if m:
            d=int(m.group(1)); nd=int(m.group(2)); h=float(m.group(3))
            data[v].setdefault(d,[]).append(h); nodes[v][d]=nd
depths = sorted(set(d for v in variants for d in data[v]))
print(f"{'depth':>5} {'nodes':>7} | {'srdw_label':>12} {'offsets_only':>13} {'nid_srdw':>10} | {'DRSW/srdw':>9} {'base/srdw':>9}")
tot={v:0.0 for v in variants}
for d in depths:
    nd = nodes.get('srdw_label',{}).get(d) or nodes.get('offsets_only',{}).get(d,0)
    vals={}
    for v in variants:
        xs=data[v].get(d,[]); vals[v]=sum(xs)/len(xs) if xs else float('nan')
        if xs: tot[v]+=sum(xs)/len(xs)
    def r(a,b):
        try: return f"{a/b:.2f}x"
        except: return "-"
    print(f"{d:>5} {nd:>7} | {vals['srdw_label']:>12.4f} {vals['offsets_only']:>13.4f} {vals['nid_srdw']:>10.4f} | {r(vals['offsets_only'],vals['srdw_label']):>9} {r(vals['nid_srdw'],vals['srdw_label']):>9}")
print("-"*80)
print(f"{'SUM':>5} {'':>7} | {tot['srdw_label']:>12.4f} {tot['offsets_only']:>13.4f} {tot['nid_srdw']:>10.4f}")
PY

echo
echo "=== correctness: per-depth node counts must match across variants (tree 0) ==="
python3 -I - <<'PY'
import re
def nodeseq(path):
    seq=[]
    seen=set()
    for line in open(path):
        m=re.search(r'^D(\d+)\s+nodes=(\d+)',line)
        if m:
            d=int(m.group(1))
            if d in seen: break   # stop at start of tree 1
            seen.add(d); seq.append((d,int(m.group(2))))
    return seq
a=nodeseq("/tmp/run_srdw_label.err")
b=nodeseq("/tmp/run_nid_srdw.err")
print("srdw_label vs nid_srdw tree0 node counts:", "MATCH" if a==b else "MISMATCH")
if a!=b:
    print("srdw_label:",a); print("nid_srdw :",b)
PY
