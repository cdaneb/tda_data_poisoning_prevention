# Preregistered downstream-classifier results

Registration: https://osf.io/h863t/

Frozen preregistration hash: `3451b9eeb5375434ad6ebaeb8273b32eb6a308fc40cdf5ff05f299de97105ade`

## Primary UNSW analysis

| Contrast | Mean | 95% hierarchical bootstrap CI | Gate |
|---|---:|---:|:---:|
| Clean - poisoned malicious recall | 0.017385 | [0.014232, 0.020409] | FAIL |
| IF60 - poisoned malicious recall | 0.008278 | [0.006708, 0.009708] | FAIL |
| IF60 - random malicious recall | 0.006747 | [0.005028, 0.008160] | PASS |
| IF60 - clean benign FPR | -0.011059 | [-0.014195, -0.008276] | PASS |

All four primary gates: **FAIL**.

The registered decision rule does not permit secondary analyses to rescue a failed primary claim.
