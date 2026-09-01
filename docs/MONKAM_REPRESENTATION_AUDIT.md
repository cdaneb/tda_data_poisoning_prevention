# Faithful Monkam representation audit

Status: complete. The locked eight-row technical pilot and full supplied-workbook audit passed shape, finite-value, deterministic-hash, shared-fit, and state-capture gates.

The notebook-operational and supplied 126-feature representation is a `1 x 1500` raster, threshold 0.3, seven filtrations, H0/H1, and nine summaries per homology dimension: `7 x 2 x 9 = 126`. Algorithm 1's conflicting threshold-0.4 control has the same arithmetic. The project control uses a `30 x 50` raster, five filtrations and six summaries: `5 x 2 x 6 = 60`. The supplied 280 workbook cell uses seven filtrations, H0/H1 and 20 summaries: `7 x 2 x 20 = 280`; that recoverable cell does not include `Scaler`.

Feature ordering is filtration-major, extractor-major, then homology-major. H0 and H1 therefore alternate within every extractor rather than occupying contiguous half-blocks. A synthetic-diagram regression test proves each documented coordinate against the corresponding standalone extractor.

On the shared-fit pilot, all `1 x 1500` configurations had identically constant H1 coordinates. Threshold 0.3 produced 66 constant dimensions; threshold 0.4 produced 63. The `30 x 50` control had nontrivial H1 and zero constant dimensions in the pilot. The fitted maximum was 255, giving byte cuts 76.5 and 102.0. Representative branch Scaler states were 0.5 for `1 x 1500` and 15.0 for `30 x 50`.

The full 280 workbook has 79,881 rows, 140 constant features (133 all-zero), 24,925 unique vectors, 59,167 repeated members (74.069%), 54,956 redundant rows (68.797%), and 22,632 conflicting-label repeated members (28.332%). The 126 workbook has 31,000 rows, 63 constant features, 7,670 unique vectors, 26,470 repeated members (85.387%), and 23,330 redundant rows (75.258%); it has no labels. The existing deep audit verified the 280 workbook's row/label alignment with the raw UNSW table. The 126 workbook cannot be row-aligned because its 10,000 attack rows were sampled without a seed.

Raw-payload, binary-mask, diagram, and final-vector equivalence profiles and hashes are recorded per pilot configuration in `results/monkam_representation_audit.json`. Separate fitting remains only the completed historical 18-poison sensitivity: it changed learned states and poison vectors but not DBSCAN assignments or zero poison removal.

Unrecoverable: the missing 1,000 poison examples and their generator/objective/seeds; selected HDBSCAN, OPTICS, and Mean Shift settings; feature reduction; and the unseeded 126-workbook sample membership. The 18 delivered poisons are a diagnostic sample from 11 parents and are not substituted for the missing population.
