#!/usr/bin/env python3
"""Weight streaming and background writes for forward.py.

ShardMap      byte offsets of every tensor, parsed from the safetensors headers.
Prefetcher    a reader thread that, in the order the forward pass consumes them, reads each layer's
              dense tensors (one GPU buffer per layer) and its routed experts (chunks of `chunk`
              experts into a fixed ring of GPU buffers) with pread into pinned staging buffers and
              copies them to the GPU on a side stream. The forward pass waits on CUDA events only; the
              bytes are the ones safetensors returns, so the math is unchanged.
LayerCacheResolver  optional: read shards from a RAM copy kept by layer_cache.py (one virtiofs read
              per shard for all concurrent processes instead of one per process).
AsyncWriter   one background thread running write jobs in submission order, bounded queue.
"""
import json
import os
import queue
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
             "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
             "BOOL": torch.bool}
ST_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ShardMap:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.wm = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self._hdr = {}
        self.by_layer = {}
        for n in self.wm:
            m = LAYER_RE.search(n)
            if m:
                self.by_layer.setdefault(int(m.group(1)), []).append(n)

    def header(self, shard):
        if shard not in self._hdr:
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            base = 8 + n
            self._hdr[shard] = {k: (v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0], base + v["data_offsets"][1])
                                for k, v in h.items() if k != "__metadata__"}
        return self._hdr[shard]

    def info(self, name):
        shard = self.wm[name]
        return (shard,) + self.header(shard)[name]

    def layer_shards(self, il):
        return sorted({self.wm[n] for n in self.by_layer.get(il, [])})


class Job:
    """A set of tensors read as contiguous byte segments of one shard. tensors: name -> (seg, off, dtype, shape)."""
    def __init__(self, key, shard, segs, tensors, ring, layer=None):
        self.key, self.shard, self.segs, self.tensors, self.ring, self.layer = key, shard, segs, tensors, ring, layer
        self.nbytes = sum(b - a for a, b in segs)


def make_job(smap, key, names, ring, layer=None, max_gap=1 << 20):
    by_shard = {}
    for n in names:
        sh, dt, shp, a, b = smap.info(n)
        by_shard.setdefault(sh, []).append((a, b, n, dt, shp))
    jobs = []
    for sh, items in sorted(by_shard.items()):
        items.sort()
        segs, tensors = [], {}
        for a, b, n, dt, shp in items:
            if segs and a - segs[-1][1] <= max_gap and a >= segs[-1][1]:
                segs[-1][1] = b
            else:
                segs.append([a, b])
            tensors[n] = (len(segs) - 1, a, dt, shp)
        segs = [tuple(s) for s in segs]
        tensors = {n: (si, a - segs[si][0], dt, shp) for n, (si, a, dt, shp) in tensors.items()}
        jobs.append(Job(key if len(by_shard) == 1 else (key, sh), sh, segs, tensors, ring, layer))
    return jobs


class Handle:
    """Tensors of one job on the GPU (views into the job's buffers)."""
    def __init__(self, job, bufs, slot):
        self.job, self.bufs, self.slot = job, bufs, slot

    def t(self, name):
        si, off, dt, shp = self.job.tensors[name]
        n = ST_SIZE[dt]
        for d in shp:
            n *= d
        raw = self.bufs[si][off:off + n]
        if dt == "U8":
            return raw.view(shp)
        if off % ST_SIZE[dt]:
            raw = raw.clone()
        return raw.view(ST_DTYPES[dt]).view(shp)

    def has(self, name):
        return name in self.job.tensors


class LayerCacheResolver:
    """Maps a shard to its RAM copy written by layer_cache.py (complete files only: the daemon
    writes <shard>.tmp and renames). Waits up to `wait` seconds for the copy, then falls back to the
    model directory. Publishes this process's current layer so the daemon knows what to keep."""
    def __init__(self, model_dir, cache_dir, tag, wait=900.0):
        self.model_dir, self.cache_dir, self.tag, self.wait = Path(model_dir), Path(cache_dir), tag, wait
        (self.cache_dir / "progress").mkdir(parents=True, exist_ok=True)
        self.prog = self.cache_dir / "progress" / f"{tag}.json"
        self.fallbacks = 0

    def progress(self, layer, done=False):
        tmp = self.prog.with_suffix(".tmp")
        json.dump({"pid": os.getpid(), "layer": int(layer), "done": bool(done), "time": time.time()}, open(tmp, "w"))
        os.replace(tmp, self.prog)

    def path(self, shard):
        p = self.cache_dir / shard
        t0, warned = time.time(), False
        while not p.exists():
            if time.time() - t0 > self.wait:
                self.fallbacks += 1
                log(f"layer cache: {shard} not cached after {self.wait:.0f}s, reading the model dir")
                return self.model_dir / shard
            if not warned and time.time() - t0 > 30:
                log(f"layer cache: waiting for {shard}")
                warned = True
            time.sleep(0.2)
        return p


class Prefetcher:
    """Reads jobs in plan order on a background thread. Ring jobs (experts) go into a fixed ring of
    `n_slots` GPU buffers of `slot_bytes`; other jobs get a fresh GPU buffer per segment. The
    consumer calls get(key) in plan order and release(handle) when its work on the handle is
    enqueued; ring slots are reused only after that work has executed (CUDA event)."""
    def __init__(self, smap, dev, plan, n_slots, slot_bytes, resolver=None, staging_mb=256, io_threads=4,
                 piece_mb=32):
        self.smap, self.dev, self.plan, self.resolver = smap, dev, plan, resolver
        self.cs = torch.cuda.Stream(device=dev)
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=dev) for _ in range(n_slots)]
        self.slot_ev = [None] * n_slots
        self.free = queue.Queue()
        for i in range(n_slots):
            self.free.put(i)
        self.ready = queue.Queue(maxsize=max(4, n_slots + 2))
        self.stage = [torch.empty(staging_mb << 20, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.stage_np = [s.numpy() for s in self.stage]
        self.stage_ev = [None, None]
        self.io = ThreadPoolExecutor(io_threads)
        self.piece = piece_mb << 20
        self.fds = {}
        self.err = None
        self.stop = False
        self.cur_layer = None
        self.stats = dict(read_s=0.0, read_bytes=0, wait_slot_s=0.0, wait_get_s=0.0, h2d_wait_s=0.0)
        self.th = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self.th.start()

    def _fd(self, shard):
        if shard not in self.fds:
            p = self.resolver.path(shard) if self.resolver else self.smap.dir / shard
            self.fds[shard] = os.open(p, os.O_RDONLY)
        return self.fds[shard]

    def drop_shard(self, shard):
        fd = self.fds.pop(shard, None)
        if fd is not None:
            os.close(fd)

    def _pread(self, fd, mv, off):
        got = 0
        while got < len(mv):
            n = os.preadv(fd, [mv[got:]], off + got)
            if n <= 0:
                raise IOError(f"short read at {off + got}")
            got += n
        return got

    def _read_into(self, fd, dst, a, b):
        """bytes [a, b) of fd into GPU tensor dst[0 : b-a] through the pinned staging buffers."""
        S = self.stage[0].numel()
        k = 0
        for p0 in range(a, b, S):
            p1 = min(b, p0 + S)
            if self.stage_ev[k] is not None:
                t = time.time(); self.stage_ev[k].synchronize(); self.stats["h2d_wait_s"] += time.time() - t
            mv = memoryview(self.stage_np[k])
            t = time.time()
            futs = [self.io.submit(self._pread, fd, mv[q0 - p0:min(p1, q0 + self.piece) - p0], q0)
                    for q0 in range(p0, p1, self.piece)]
            for f in futs:
                f.result()
            self.stats["read_s"] += time.time() - t
            self.stats["read_bytes"] += p1 - p0
            with torch.cuda.stream(self.cs):
                dst[p0 - a:p1 - a].copy_(self.stage[k][:p1 - p0], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.cs)
            self.stage_ev[k] = ev
            k ^= 1

    def _run(self):
        try:
            torch.cuda.set_device(self.dev)
            prev = None
            for job in self.plan:
                if self.stop:
                    break
                if prev is not None and job.shard != prev:
                    self.drop_shard(prev)     # an open fd would pin a deleted RAM-cache file
                prev = job.shard
                if self.resolver is not None and job.layer is not None and job.layer != self.cur_layer:
                    self.cur_layer = job.layer
                    self.resolver.progress(job.layer)
                fd = self._fd(job.shard)
                slot = None
                if job.ring:
                    t = time.time()
                    slot = self.free.get()
                    self.stats["wait_slot_s"] += time.time() - t
                    if self.slot_ev[slot] is not None:
                        self.cs.wait_event(self.slot_ev[slot])
                    assert job.nbytes <= self.slots[slot].numel(), (job.key, job.nbytes)
                    bufs, o = [], 0
                    for a, b in job.segs:                 # segments packed back to back in the slot
                        bufs.append(self.slots[slot][o:o + b - a])
                        o += b - a
                    for (a, b), buf in zip(job.segs, bufs):
                        self._read_into(fd, buf, a, b)
                else:
                    with torch.cuda.stream(self.cs):
                        bufs = [torch.empty(b - a, dtype=torch.uint8, device=self.dev) for a, b in job.segs]
                    for (a, b), buf in zip(job.segs, bufs):
                        self._read_into(fd, buf, a, b)
                with torch.cuda.stream(self.cs):
                    done = torch.cuda.Event()
                    done.record(self.cs)
                self.ready.put((job, bufs, slot, done))
        except BaseException as e:      # surfaced to the consumer in get()
            self.err = e
            self.ready.put(None)

    def get(self, key):
        t = time.time()
        item = self.ready.get()
        self.stats["wait_get_s"] += time.time() - t
        if item is None:
            raise RuntimeError(f"prefetch thread failed: {self.err!r}") from self.err
        job, bufs, slot, done = item
        assert job.key == key, f"prefetch order mismatch: wanted {key}, got {job.key}"
        cur = torch.cuda.current_stream(self.dev)
        cur.wait_event(done)
        if slot is None:
            for b in bufs:
                b.record_stream(cur)
        return Handle(job, bufs, slot)

    def release(self, h):
        if h.slot is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev))
            self.slot_ev[h.slot] = ev
            self.free.put(h.slot)
        h.bufs = None

    def close(self):
        self.stop = True
        while self.th.is_alive():
            try:
                self.ready.get(timeout=0.1)       # unblock a reader waiting on a full queue
            except queue.Empty:
                pass
            try:
                self.free.put_nowait(0)
            except queue.Full:
                pass
            self.th.join(timeout=0.1)
        self.io.shutdown()
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}


def build_plan(smap, layers, dense_lead, n_experts, chunk, pfx):
    """Jobs in consumption order: per layer its dense tensors, then (MoE layers) expert chunks."""
    for il in layers:
        names = smap.by_layer.get(il, [])
        dense = [n for n in names if not EXPERT_RE.search(n)]
        for j in make_job(smap, ("dense", il), dense, ring=False, layer=il):
            yield j
        if il < dense_lead:
            continue
        ep = f"{pfx}layers.{il}.block_sparse_moe.experts."
        for c0 in range(0, n_experts, chunk):
            nm = [f"{ep}{e}.{w}.{s}" for e in range(c0, min(n_experts, c0 + chunk))
                  for w in ("w1", "w2", "w3") for s in ("weight_packed", "weight_scale")]
            js = make_job(smap, ("exp", il, c0 // chunk), nm, ring=True, layer=il)
            assert len(js) == 1, f"layer {il} experts {c0}.. span several shards"
            yield js[0]


def expert_chunk_bytes(smap, il, n_experts, chunk, pfx):
    ep = f"{pfx}layers.{il}.block_sparse_moe.experts."
    best = 0
    for c0 in range(0, n_experts, chunk):
        nm = [f"{ep}{e}.{w}.{s}" for e in range(c0, min(n_experts, c0 + chunk))
              for w in ("w1", "w2", "w3") for s in ("weight_packed", "weight_scale")]
        best = max(best, make_job(smap, None, nm, ring=True)[0].nbytes)
    return best


class AsyncWriter:
    """Runs write jobs (callables) on one thread in submission order. submit() blocks while
    `depth` jobs are pending; drain() waits for all; errors re-raise in the caller."""
    def __init__(self, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.err = None
        self.busy_s = 0.0
        self.th = threading.Thread(target=self._run, name="writer", daemon=True)
        self.th.start()

    def _run(self):
        while True:
            fn = self.q.get()
            if fn is None:
                self.q.task_done()
                return
            t = time.time()
            try:
                if self.err is None:
                    fn()
            except BaseException as e:
                self.err = e
                log(f"writer job failed: {e!r}")
            self.busy_s += time.time() - t
            self.q.task_done()

    def _check(self):
        if self.err is not None:
            raise RuntimeError(f"background write failed: {self.err!r}") from self.err

    def submit(self, fn):
        self._check()
        self.q.put(fn)

    def drain(self):
        self.q.join()
        self._check()

    def close(self):
        self.q.put(None)
        self.th.join()
        self._check()
