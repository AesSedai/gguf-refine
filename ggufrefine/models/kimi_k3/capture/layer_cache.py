#!/usr/bin/env python3
"""RAM copy of the model shards that concurrent capture processes are about to read.

virtiofs serves every process its own read (no guest page cache: 8 readers of one file are not
faster than 8 readers of 8 files), so 8 processes streaming a 17 GB layer shard read 136 GB per
layer step. This daemon reads each layer shard once into --cache-dir (tmpfs, e.g. /dev/shm) and the
workers (forward.py --layer-cache DIR) read it from there.

Workers publish the layer their weight reader is on in <cache-dir>/progress/<tag>.json (pid, layer,
done). The daemon keeps the shards of layers [base, base + --window), base = the lowest layer of any
live, unfinished worker; shards of layers below base are deleted. Copies are written as <shard>.tmp
and renamed when complete, so a worker never sees a partial file. Total size is capped by --max-gb.

    python -m ggufrefine.models.kimi_k3.capture.layer_cache --model M --cache-dir /dev/shm/k31-layer-cache --window 3 --exit-when-done
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .stream import ShardMap, log  # noqa: E402


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def copy_file(src, dst, threads, piece=64 << 20):
    size = src.stat().st_size
    tmp = dst.with_name(dst.name + ".tmp")
    fi = os.open(src, os.O_RDONLY)
    fo = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fo, size)

        def one(off):
            buf = bytearray(min(piece, size - off))
            mv = memoryview(buf)
            got = 0
            while got < len(buf):
                n = os.preadv(fi, [mv[got:]], off + got)
                if n <= 0:
                    raise IOError(f"{src}: short read at {off + got}")
                got += n
            put = 0
            while put < len(buf):
                put += os.pwrite(fo, mv[put:], off + put)
            return len(buf)

        with ThreadPoolExecutor(threads) as ex:
            total = sum(ex.map(one, range(0, size, piece)))
        assert total == size, (total, size)
    finally:
        os.close(fi)
        os.close(fo)
    os.replace(tmp, dst)
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=3, help="layers kept from the slowest worker's layer on")
    ap.add_argument("--max-gb", type=float, default=64.0)
    ap.add_argument("--layers", type=int, default=None, help="number of layers (default: config)")
    ap.add_argument("--start-layer", type=int, default=0, help="base layer before any worker reports")
    ap.add_argument("--threads", type=int, default=8, help="read threads per shard copy")
    ap.add_argument("--parallel", type=int, default=1, help="shards copied concurrently (measured with 8 workers "
                    "also writing dumps: 1 copy 1.4 GB/s, 2-3 copies 0.5-1.1 GB/s each; keep 1)")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--exit-when-done", action="store_true",
                    help="exit (and empty the cache) once every worker that reported is done or dead")
    a = ap.parse_args()
    smap = ShardMap(a.model)
    n_layer = a.layers or json.load(open(a.model / "config.json"))["text_config"]["num_hidden_layers"]
    shard_layers = {}
    for il in range(n_layer):
        for sh in smap.layer_shards(il):
            shard_layers.setdefault(sh, []).append(il)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    (a.cache_dir / "progress").mkdir(exist_ok=True)
    for f in list(a.cache_dir.glob("*.tmp")) + list((a.cache_dir / "progress").glob("*.json")):
        f.unlink()                                   # partial copies and progress of earlier runs
    log(f"layer cache: {a.cache_dir}, window {a.window} layers, cap {a.max_gb} GB, {n_layer} layers")
    t_last, copied, copy_s, seen_any = time.time(), 0, 0.0, False
    pool = ThreadPoolExecutor(a.parallel)
    inflight = {}                                     # shard -> (future, size, t0)
    while True:
        for sh in [s for s, (f, _, _) in inflight.items() if f.done()]:
            f, sz, t = inflight.pop(sh)
            f.result()                                # raise copy errors
            dt = time.time() - t
            copied += sz; copy_s += dt
            log(f"layer cache: {sh} (layers {shard_layers[sh][0]}-{shard_layers[sh][-1]}) {sz / 1e9:.1f} GB in {dt:.1f}s "
                f"({sz / 1e9 / max(dt, 1e-9):.2f} GB/s), {len(inflight)} copies in flight")
        workers = []
        for f in (a.cache_dir / "progress").glob("*.json"):
            try:
                workers.append(json.load(open(f)))
            except (json.JSONDecodeError, OSError):
                pass                                  # being replaced
        live = [w for w in workers if not w.get("done") and alive(w["pid"])]
        seen_any = seen_any or bool(workers)
        if seen_any and not live and not inflight:
            if a.exit_when_done:
                for sh in shard_layers:
                    (a.cache_dir / sh).unlink(missing_ok=True)
                log(f"layer cache: all {len(workers)} workers finished; cache emptied. "
                    f"copied {copied / 1e9:.0f} GB in {copy_s:.0f}s")
                return 0
            time.sleep(a.poll)
            continue
        if seen_any and not live:                     # workers finished: no new copies, wait for in-flight ones
            time.sleep(a.poll)
            continue
        base = min(w["layer"] for w in live) if live else a.start_layer
        want = []
        for il in range(base, min(n_layer, base + a.window)):
            for sh in smap.layer_shards(il):
                if sh not in want:
                    want.append(sh)
        for sh, ls in shard_layers.items():          # behind every live worker
            if max(ls) < base and (a.cache_dir / sh).exists() and sh not in inflight:
                (a.cache_dir / sh).unlink()
        used = sum((a.cache_dir / sh).stat().st_size for sh in shard_layers if (a.cache_dir / sh).exists())
        used += sum(sz for _, sz, _ in inflight.values())
        for sh in want:                              # in layer order, nearest first
            if len(inflight) >= a.parallel:
                break
            dst = a.cache_dir / sh
            if dst.exists() or sh in inflight:
                continue
            sz = (a.model / sh).stat().st_size
            if used + sz > a.max_gb * 1e9:
                break
            inflight[sh] = (pool.submit(copy_file, a.model / sh, dst, a.threads), sz, time.time())
            used += sz
        if time.time() - t_last > 60:
            log(f"layer cache: base layer {base}, {len(live)} live workers, cache {used / 1e9:.0f} GB, "
                f"{len(inflight)} copies in flight, {copied / 1e9:.0f} GB copied")
            t_last = time.time()
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
