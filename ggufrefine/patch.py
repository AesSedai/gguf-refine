"""Overwrite tensor data inside a GGUF in place, with a backup for revert.

Only same-size replacements (no offset in the file moves). The backup keeps the original bytes plus a manifest
with sha256 of original and patched data; apply refuses to overwrite an existing backup of the same tensor, so a
revert always restores the bytes that were there before the first apply.

    python -m ggufrefine.patch apply  --gguf X.gguf --backup DIR NAME=raw.bin [NAME=raw.bin ...]
    python -m ggufrefine.patch revert --gguf X.gguf --backup DIR [NAME ...]        (default: all in DIR)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from . import ggufio


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _save(man_path, man):
    tmp = man_path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, man_path)


def apply(gguf, backup, items):
    """items: [(tensor name, path of raw bytes)]"""
    backup = Path(backup)
    backup.mkdir(parents=True, exist_ok=True)
    man_path = backup / "manifest.json"
    man = json.load(open(man_path)) if man_path.exists() else {}
    info = ggufio.tensor_table(gguf)
    with open(gguf, "r+b") as f:
        for name, src in items:
            ti = info[name]
            new = Path(src).read_bytes()
            if len(new) != ti.nbytes:
                raise ValueError(f"{name}: {len(new)} bytes given, tensor holds {ti.nbytes} ({ti.type} {ti.shape})")
            if name in man:
                raise ValueError(f"{name}: already patched (backup exists) - revert first")
            f.seek(ti.offset); orig = f.read(ti.nbytes)
            (backup / f"{name}.orig").write_bytes(orig)
            f.seek(ti.offset); f.write(new); f.flush(); os.fsync(f.fileno())
            f.seek(ti.offset)
            if f.read(ti.nbytes) != new:
                raise IOError(f"{name}: read-back mismatch after write")
            man[name] = {"offset": ti.offset, "bytes": ti.nbytes, "type": ti.type, "shape": list(ti.shape),
                         "sha256_orig": _sha(orig), "sha256_patched": _sha(new), "source": str(Path(src).resolve())}
            _save(man_path, man)
            print(f"patched {name} ({ti.type}, {ti.nbytes / 2**20:.1f} MiB) at offset {ti.offset}", flush=True)


def revert(gguf, backup, names=None):
    backup = Path(backup)
    man_path = backup / "manifest.json"
    man = json.load(open(man_path)) if man_path.exists() else {}
    with open(gguf, "r+b") as f:
        for name in (names or list(man)):
            m = man[name]
            orig = (backup / f"{name}.orig").read_bytes()
            if _sha(orig) != m["sha256_orig"]:
                raise IOError(f"{name}: backup is corrupt")
            f.seek(m["offset"])
            if _sha(f.read(m["bytes"])) != m["sha256_patched"]:
                print(f"warning: {name} does not hold the patched bytes any more; restoring the original anyway")
            f.seek(m["offset"]); f.write(orig); f.flush(); os.fsync(f.fileno())
            (backup / f"{name}.orig").unlink()
            del man[name]
            _save(man_path, man)
            print(f"reverted {name}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply", "revert"])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--backup", required=True, type=Path)
    ap.add_argument("items", nargs="*")
    a = ap.parse_args(argv)
    if a.action == "apply":
        apply(a.gguf, a.backup, [tuple(it.split("=", 1)) for it in a.items])
    else:
        revert(a.gguf, a.backup, a.items or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
