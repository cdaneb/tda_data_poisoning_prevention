# Duplicate-safe replication of supplied classifier results

Status: complete, five fixed seeds. The primary population retains all 79,881 aligned observations and all conflicting labels; nothing was silently deduplicated.

The notebook protocol uses unscaled raw 1,500-byte or supplied 280-feature inputs, default `RandomForestClassifier`, random state 60 for splitting, test size 0.2 for the ten-class task and 0.1 for normal-versus-malicious. The notebook leaves the forest seed unset; the controlled comparison fixes each forest to its split seed and uses eight workers.

Five-seed accuracy (population SD) was:

| Task | Split | Raw | TDA-280 |
|---|---|---:|---:|
| Multiclass | random row | 0.8452 (0.0026) | 0.7958 (0.0029) |
| Multiclass | raw-payload groups | 0.8062 (0.0162) | 0.7583 (0.0157) |
| Multiclass | TDA-vector groups | 0.8030 (0.0736) | 0.7293 (0.1034) |
| Binary | random row | 0.9924 (0.0007) | 0.9581 (0.0015) |
| Binary | raw-payload groups | 0.9666 (0.0124) | 0.9388 (0.0178) |
| Binary | TDA-vector groups | 0.9080 (0.0616) | 0.8688 (0.0888) |

Thus the saved random-row scores are reproducible (saved values were about 0.8420/0.7922 multiclass and 0.9920/0.9581 binary), but random-row splitting materially inflates performance. Relative to random rows, TDA accuracy falls 3.75 points under raw-payload grouping and 6.65 points under final-vector grouping for multiclass; binary falls 1.93 and 8.93 points. Variability also rises sharply under final-vector grouping because large conflicting-label equivalence classes constrain class composition.

Random-row splits averaged thousands of cross-split identity classes. Raw grouping makes raw overlap exactly zero but leaves mean TDA overlap of 639 multiclass and 404 binary classes. TDA-vector grouping makes both overlaps zero. Full macro/weighted F1, balanced accuracy, per-class precision/recall, confusion matrices, novel/repeated performance, class counts, split hashes, uncertainty intervals, and exclusions are in the merged JSON and per-cell artifacts.
