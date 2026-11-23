#!/usr/bin/env python3
"""Sum per-process dense Grams (format in ggufrefine.capture.grams) into splits.

    src:  <src>/<partdir>/layer-<L>/{meta.json, <group>.tri.f32, <group>.sum.f64}   (one partdir per process)
    out:  <out>/<split>/layer-<L>/{meta.json, <group>.tri.f32, <group>.sum.f64}    same file format, plus
          meta "parts": the part ids summed in; <out>/index.json lists splits, layers, groups, counts.

Splits by the part id in each source meta: test = --test-parts (default 29,30), train = every other
part. Merging is incremental and idempotent: a (split, layer) already holding a part skips it, so
the merger can run after every round (or continuously, --watch) and --delete removes each source
layer directory once its sums are in place. Per (split, layer) and pass:
    acc(fp64) = merged(fp32 -> fp64) + sum over new parts (fp32 -> fp64)
    write layer-L.new/, re-read every file and compare with acc (exact), count check, then swap
    layer-L.new -> layer-L (old copy removed after the rename). Only then are sources deleted.
A crash leaves either the old or the new layer directory complete; layer-L.new is discarded and
an orphaned layer-L.old restored on the next start.

    python -m ggufrefine.capture.merge_grams --src /dev/shm/grams --out OUT/grams --delete                 # one pass
    python -m ggufrefine.capture.merge_grams --src /dev/shm/grams --out OUT/grams --delete --watch --until-file F
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .grams import FORMAT, VERSION, tri_len  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lkey(name):
    return (1, 0) if name == "output" else (0, int(name))


def read_meta(d):
    m = json.load(open(d / "meta.json"))
    assert m["format"] == FORMAT and m["version"] == VERSION, (d, m.get("format"), m.get("version"))
    return m


def load(d, files, info, dt_tri=np.float32):
    K, H = info["K"], info["heads"]
    tshape = (tri_len(K),) if H is None else (H, tri_len(K))
    sshape = (K,) if H is None else (H, K)
    tri = np.fromfile(d / files["tri"], dtype=dt_tri)
    s = np.fromfile(d / files["sum"], dtype=np.float64)
    if tri.size != int(np.prod(tshape)) or s.size != int(np.prod(sshape)):
        raise ValueError(f"{d}: {files}: sizes {tri.size}/{s.size}, expected {tshape}/{sshape}")
    return tri.reshape(tshape), s.reshape(sshape)


# layer dir names are constructed, not listed: readdir on the virtiofs mount fails with ENOMEM at times
LAYER_NAMES = [str(i) for i in range(256)] + ["output"]


def recover(out):
    for split in ("train", "test"):
        for nm in LAYER_NAMES:
            d = out / split / f"layer-{nm}.new"
            if d.exists():
                shutil.rmtree(d)
                log(f"recover: discarded unfinished {d}")
            d = out / split / f"layer-{nm}.old"
            if not d.exists():
                continue
            live = d.with_name(d.name[:-4])
            if live.exists():
                shutil.rmtree(d)
            else:
                os.replace(d, live)
                log(f"recover: restored {live}")


def merge_layer(out, split, layer, srcs):
    """srcs: [(part_id, dir, meta)] all of this split and layer. Returns the part ids added."""
    dst = out / split / f"layer-{layer}"
    old = read_meta(dst) if (dst / "meta.json").exists() else None
    have = set(old["parts"]) if old else set()
    new = [(p, d, m) for p, d, m in srcs if p not in have]
    if not new:
        return []
    ref_groups = old["groups"] if old else new[0][2]["groups"]
    acc = {}
    for g, info in ref_groups.items():
        if old:
            t, s = load(dst, info["files"], info)
            acc[g] = [t.astype(np.float64), s.copy(), int(info["n"])]
        else:
            K, H = info["K"], info["heads"]
            acc[g] = [np.zeros((tri_len(K),) if H is None else (H, tri_len(K))), np.zeros((K,) if H is None else (H, K)), 0]
    for p, d, m in new:
        if set(m["groups"]) != set(ref_groups):
            raise ValueError(f"{d}: groups {sorted(m['groups'])} != {sorted(ref_groups)}")
        for g, info in m["groups"].items():
            r = ref_groups[g]
            if (info["K"], info["heads"], info["weights"]) != (r["K"], r["heads"], r["weights"]):
                raise ValueError(f"{d}: group {g} shape/weights differ from the merged ones")
            t, s = load(d, info["files"], info)
            acc[g][0] += t
            acc[g][1] += s
            acc[g][2] += int(info["n"])
    tmp = dst.with_name(dst.name + ".new")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    groups, arrs = {}, {}
    for g, (t, s, n) in acc.items():
        r = ref_groups[g]
        files = {"tri": f"{g}.tri.f32", "sum": f"{g}.sum.f64"}
        t32 = t.astype(np.float32)
        for arr, fn in ((t32, files["tri"]), (s, files["sum"])):
            with open(tmp / fn, "wb") as f:
                f.write(np.ascontiguousarray(arr).tobytes())
                f.flush()
                os.fsync(f.fileno())
        arrs[g] = (t32, s)
        groups[g] = dict(weights=r["weights"], K=r["K"], heads=r["heads"], n=n, files=files,
                         tri_shape=list(t32.shape), sum_shape=list(s.shape))
    parts = sorted(have | {p for p, _, _ in new})
    first = new[0][2]
    meta = dict(format=FORMAT, version=VERSION, layer=layer, split=split, parts=parts, groups=groups,
                layout=first["layout"], n_ctx=first["n_ctx"], tokens_file=first["tokens_file"],
                fast=sorted({bool(m.get("fast")) for _, _, m in new} | set((old or {}).get("fast", []))),
                merged=time.strftime("%Y-%m-%d %H:%M:%S"),
                storage="tri float32 (fp64 sums rounded once per merge pass), sum float64")
    json.dump(meta, open(tmp / "meta.json", "w"), indent=1)
    # verify by re-reading what was written
    m2 = read_meta(tmp)
    for g, info in m2["groups"].items():
        t, s = load(tmp, info["files"], info)
        if not (np.array_equal(t, arrs[g][0]) and np.array_equal(s, arrs[g][1])):
            raise IOError(f"{tmp}: group {g} re-read differs from the sums")
        want_n = (int(old["groups"][g]["n"]) if old else 0) + sum(int(m["groups"][g]["n"]) for _, _, m in new)
        if info["n"] != want_n:
            raise IOError(f"{tmp}: group {g} count {info['n']} != {want_n}")
    if dst.exists():
        bak = dst.with_name(dst.name + ".old")
        os.replace(dst, bak)
        os.replace(tmp, dst)
        shutil.rmtree(bak)
    else:
        os.replace(tmp, dst)
    return [p for p, _, _ in new]


def scan(src):
    """complete source layer dirs: {layer: [(part, dir, meta)]}"""
    out = {}
    for pd in sorted(src.iterdir()) if src.exists() else []:
        if not pd.is_dir():
            continue
        for d in pd.glob("layer-*"):
            if not (d / "meta.json").exists():
                continue
            m = read_meta(d)
            if m.get("part") is None:
                raise ValueError(f"{d}: no part id in meta (capture with --part-id)")
            prev = [x for x in out.get(str(m["layer"]), []) if x[0] == int(m["part"])]
            if prev:
                raise SystemExit(f"layer {m['layer']}: part {m['part']} in both {prev[0][1]} and {d}")
            out.setdefault(str(m["layer"]), []).append((int(m["part"]), d, m))
    return out


def write_index(out, test_parts, check=None):
    """out/index.json from the merged layers; check(idx) runs before it is published (raises to refuse)"""
    idx = {"format": FORMAT + "-index", "version": VERSION,
           "splits": {"test": {"definition": f"parts {sorted(test_parts)}"},
                      "train": {"definition": f"every part except {sorted(test_parts)}"}},
           "storage": "per layer dir: <group>.tri.f32 packed row-major upper triangle (row i = G[i, i:]; "
                      "heads: H triangles back to back), <group>.sum.f64; G = sum_t x x^T over the split's tokens; "
                      "ggufrefine.capture.grams.unpack_tri restores [.., K, K]",
           "layers": {}}
    total = 0
    for split in ("train", "test"):
        parts = set()
        for nm in LAYER_NAMES:
            d = out / split / f"layer-{nm}"
            if not (d / "meta.json").exists():
                continue
            m = read_meta(d)
            parts |= set(m["parts"])
            ent = idx["layers"].setdefault(str(m["layer"]), {})
            ent[split] = {"parts": m["parts"], "dir": f"{split}/{d.name}",
                          "groups": {g: {k: v[k] for k in ("weights", "K", "heads", "n", "files")}
                                     for g, v in m["groups"].items()}}
            total += sum((d / f).stat().st_size for v in m["groups"].values() for f in v["files"].values())
        idx["splits"][split]["parts"] = sorted(parts)
    idx["bytes"] = total
    idx["layers"] = dict(sorted(idx["layers"].items(), key=lambda kv: lkey(kv[0])))
    if check is not None:
        check(idx)
    tmp = out / "index.json.tmp"
    json.dump(idx, open(tmp, "w"), indent=1)
    os.replace(tmp, out / "index.json")
    return total


def one_pass(a, test_parts, final):
    found = scan(a.src)
    jobs = []
    now = time.time()
    for layer, srcs in found.items():
        for split in ("train", "test"):
            ss = [x for x in srcs if (x[0] in test_parts) == (split == "test")]
            if not ss:
                continue
            if a.expect and not final:          # wait for the round's other parts unless the files are old
                want = {p for p in a.expect if (p in test_parts) == (split == "test")}
                have = {p for p, _, _ in ss}
                oldest = min((d / "meta.json").stat().st_mtime for _, d, _ in ss)
                if not want <= have and now - oldest < a.max_wait:
                    continue
            jobs.append((split, layer, ss))
    if not jobs:
        return 0

    def run(job):
        split, layer, ss = job
        t = time.time()
        added = merge_layer(a.out, split, layer, ss)
        if a.delete:
            for p, d, _ in ss:                  # merged now or before: the sums are in place
                shutil.rmtree(d)
        return split, layer, added, [p for p, _, _ in ss if p not in added], time.time() - t

    n = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for split, layer, added, skipped, dt in ex.map(run, sorted(jobs, key=lambda j: lkey(j[1]))):
            n += 1
            log(f"{split} layer {layer}: +parts {added}{f' (already merged: {skipped})' if skipped else ''} {dt:.1f}s")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-parts", default="29,30")
    ap.add_argument("--delete", action="store_true", help="remove source layer dirs once merged and verified")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--watch", action="store_true", help="repeat passes until --until-file exists, then a final pass")
    ap.add_argument("--until-file", type=Path, default=None)
    ap.add_argument("--expect", default="", help="with --watch: part ids of the running round; a layer is merged "
                    "once all of them (in that split) have it, or its files are older than --max-wait")
    ap.add_argument("--max-wait", type=float, default=900.0)
    ap.add_argument("--require-parts", default="", help="after the final pass, every layer must hold exactly these "
                    "part ids (train: those not in --test-parts, test: those in it); fails otherwise")
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()
    test_parts = {int(x) for x in a.test_parts.split(",") if x}
    a.expect = {int(x) for x in a.expect.split(",") if x}
    a.out.mkdir(parents=True, exist_ok=True)
    recover(a.out)
    if a.watch:
        assert a.until_file is not None, "--watch needs --until-file"
        last = 0.0
        while not a.until_file.exists():
            if one_pass(a, test_parts, final=False) == 0:
                time.sleep(a.poll)
            if time.time() - last > 300:
                write_index(a.out, test_parts)
                last = time.time()
    one_pass(a, test_parts, final=True)
    check = None
    if a.require_parts:
        req = {int(x) for x in a.require_parts.split(",") if x}
        want = {"train": sorted(req - test_parts), "test": sorted(req & test_parts)}

        def check(idx):
            bad = [f"layer {L} {s}: parts {ent.get(s, {}).get('parts')} (want {want[s]})"
                   for L, ent in idx["layers"].items() for s in want if want[s] and ent.get(s, {}).get("parts") != want[s]]
            if not idx["layers"] or bad:
                raise SystemExit("merged Grams do not hold the required parts (no index.json written):\n  "
                                 + "\n  ".join(bad[:20] or ["no layers"]))
            log(f"every layer holds train parts {want['train']} and test parts {want['test']}")
    if a.require_parts:                         # an index from an earlier, unchecked pass must not stand
        (a.out / "index.json").unlink(missing_ok=True)
    total = write_index(a.out, test_parts, check)
    left = sum(len(v) for v in scan(a.src).values())
    log(f"index written: {total / 1e9:.1f} GB merged in {a.out}; {left} source layer dirs left in {a.src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
