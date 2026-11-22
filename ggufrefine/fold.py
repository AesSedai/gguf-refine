"""Fold refined per-expert codes (NpySink output, <tensor>-e<E>.npy) into a GGUF of the same layout, in place.

A tensor is written once all of its experts' files exist: assembled in expert order, checked against the
GGUF tensor size, written with ggufio.write_verified (parallel pwrite, fsync, read-back compare), recorded in
the manifest (json lines: tensor, offset, bytes, sha256), and only then are its .npy files deleted (unless
--keep). Tensors already in the manifest are skipped, so it can be rerun or run while refinement is going.

    python -m ggufrefine.fold --gguf SUBSTRATE.gguf --npy-dir out/ --manifest logs/fold.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import ggufio


def fold(gguf, npy_dir, manifest, keep=False, threads=8):
    info = ggufio.tensor_table(gguf)
    manifest = Path(manifest)
    done = set()
    if manifest.exists():
        with open(manifest) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["tensor"])
                except json.JSONDecodeError:                                  # torn tail: that tensor is redone
                    pass
        with open(manifest, "rb+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")
    files = {}
    for p in Path(npy_dir).glob("*-e*.npy"):
        m = re.fullmatch(r"(.+)-e(\d+)\.npy", p.name)
        if m:
            files.setdefault(m.group(1), {})[int(m.group(2))] = p
    n = 0
    for name, by_e in sorted(files.items()):
        if name in done:
            continue
        ti = info[name]
        if len(by_e) < ti.n_experts:
            continue
        t0 = time.time()
        buf = np.empty((ti.n_experts, ti.expert_bytes), np.uint8)
        for e in range(ti.n_experts):
            a = np.load(by_e[e])
            assert a.dtype == np.uint8 and a.nbytes == ti.expert_bytes, (by_e[e], a.dtype, a.nbytes, ti.expert_bytes)
            buf[e] = a.reshape(-1)
        sha = ggufio.write_verified(gguf, ti.offset, buf, nbytes=ti.nbytes, threads=threads)
        with open(manifest, "a") as fh:
            fh.write(json.dumps({"tensor": name, "offset": ti.offset, "bytes": ti.nbytes, "sha256": sha,
                                 "gguf": str(Path(gguf).resolve()), "time": time.strftime("%H:%M:%S")}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())                                             # recorded before the sources are deleted
        if not keep:
            for p in by_e.values():
                p.unlink()
        n += 1
        print(f"[{time.strftime('%H:%M:%S')}] folded {name} ({ti.nbytes / 2**30:.2f} GiB) in {time.time() - t0:.0f}s", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    print(f"folded {fold(a.gguf, a.npy_dir, a.manifest, a.keep, a.threads)} tensors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
