"""Triton search kernels for ggufrefine.refine.engine, one module per sign scheme.

The IQ types differ in how a group's signs are stored, and that decides which codewords are legal, so each scheme
needs its own candidate search:

| module      | types                    | signs                                    | search per item                      |
|-------------|--------------------------|------------------------------------------|--------------------------------------|
| free_sign   | IQ2_S, IQ3_S             | every sign bit stored                    | signed codewords / magnitude + signs |
| parity_sign | IQ2_XXS, IQ2_XS, IQ3_XXS | 7 stored, the 8th implied by even parity | magnitudes, then the 128 legal signs |
| ternary     | IQ1_S, IQ1_M             | grid in {-1, 0, +1} plus a shared ±delta | grid rows (+ delta flip for IQ1_S)   |

common holds what they share: the 8-wide quadratic-score features, the fp32 acceptance gate, argmin and 8x8 form.
"""
