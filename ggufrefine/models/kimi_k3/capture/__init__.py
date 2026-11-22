"""Kimi-K3.1 activation capture: a streaming fp32 PyTorch forward (forward.py) that writes routed-expert dumps
(read by ..expert_dump), dense-matmul Grams (ggufrefine.capture.grams, merged by ggufrefine.capture.merge_grams),
imatrix partials (write_imatrix.py) and KLD-base parts (merge_kld_base.py). stream.py streams the weights,
layer_cache.py shares one RAM copy of the shards between processes, status.py reports progress from the outputs.
capture.sh runs a multi-part capture, one process per GPU."""
