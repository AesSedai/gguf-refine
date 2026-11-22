"""Where the patched llama.cpp lives (native/build.sh output) and its gguf-py.

GGUF_REFINE_LLAMA overrides the source tree (default <repo>/native/llama.cpp), GGUF_REFINE_BIN the
directory with the built libraries and tools (default <source>/build/bin). Importing this module puts
that tree's gguf-py first on sys.path, so every module reads GGUFs with the same (patched) gguf-py.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LLAMA_SRC = Path(os.environ.get("GGUF_REFINE_LLAMA", REPO / "native" / "llama.cpp"))
LLAMA_BIN = Path(os.environ.get("GGUF_REFINE_BIN", LLAMA_SRC / "build" / "bin"))

_gguf_py = LLAMA_SRC / "gguf-py"
if _gguf_py.is_dir() and str(_gguf_py) not in sys.path:
    sys.path.insert(0, str(_gguf_py))


def tool(name):
    """path of a built llama.cpp tool, e.g. tool("llama-kld-swap")"""
    p = LLAMA_BIN / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run native/build.sh or set GGUF_REFINE_BIN")
    return p
