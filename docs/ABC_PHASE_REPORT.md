LENS4 ROBUSTNESS TESTS — Final Report (A, B, C all complete)

TEST A — Binarization threshold 0.3 vs 0.4

┌─────────────────────────────────────┬────────────────────────────┬─────────────────────────────────────────┐
│                                     │   threshold=0.4 (prior)    │        threshold=0.3 (this test)        │
├─────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────┤
│ Swap count-invariance (200 samples) │ 0/200 changed              │ 0/200 changed — confirmed               │
├─────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────┤
│ Noise |Δ| mean                      │ (prior run)                │ 36.1 (foreground count ~190.6 baseline) │
├─────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────┤
│ S mean±std                          │ 1.80% ± 0.51%              │ 1.24% ± 1.02%                           │
├─────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────┤
│ SG mean±std                         │ 6.48% ± 1.24%              │ 6.48% ± 2.44%                           │
├─────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────┤
│ Guidance effect                     │ +4.68 ± 1.57, 5/5 positive │ +5.24 ± 2.35, 5/5 positive              │
└─────────────────────────────────────┴────────────────────────────┴─────────────────────────────────────────┘

Verdict: threshold-independent. Both parts of the claim survive at 0.3 unchanged in kind: the deductive swap-invariance proof holds exactly (0/200 at either threshold, as it must), and the empirical guidance effect keeps its sign and full 5/5-seed consistency, at a comparable (slightly larger, more variable) magnitude. The paper's internal 0.3-vs-0.4 ambiguity is now a demonstrated non-issue for this conclusion.

TEST B — Other multiset-preserving permutation families (threshold=0.4, no noise, no guidance, malicious-only, 5 seeds, OPTICS)

┌────────────────────────────────┬──────────────────┬──────────────────────────────┬──────────────────────┐
│             Family             │ Count-invariant? │ Mean positions value-changed │ Capture % (mean±std) │
├────────────────────────────────┼──────────────────┼──────────────────────────────┼──────────────────────┤
│ Transpositions (60 swaps, = S) │ 0/200            │ 21.96 ± 30.4                 │ 1.80% ± 0.51%        │
├────────────────────────────────┼──────────────────┼──────────────────────────────┼──────────────────────┤
│ Block reversal (k=120)         │ 0/200            │ 16.42 ± 39.5                 │ 0.00% ± 0.00%        │
├────────────────────────────────┼──────────────────┼──────────────────────────────┼──────────────────────┤
│ Block swap (2×k=60)            │ 0/200            │ 14.70 ± 37.2                 │ 0.00% ± 0.00%        │
├────────────────────────────────┼──────────────────┼──────────────────────────────┼──────────────────────┤
│ Cyclic shift (random rotation) │ 0/200            │ 266.6 ± 375.6                │ 6.28% ± 1.31%        │
└────────────────────────────────┴──────────────────┴──────────────────────────────┴──────────────────────┘

Verdict: yes — identical count-invariance, materially different capture. All four families hit exactly 0/200 on the binarizer check, confirming the proof in action across every permutation family tested, while capture spans 0.00% to 6.28%. This is direct evidence that detection of swap-family attacks flows entirely through the position-dependent (Height/Radial filtration) channel, never through binarized foreground count. The ranking tracks actual value disruption, not byte-position count touched: despite moving as many byte positions as the transposition reference, block reversal and block swap land at exactly 0% — a genuinely surprising result (contiguous relocation is less detectable here than scattered transpositions of similar footprint, not equally detectable), plausibly because UNSW payloads carry heavy zero-padding that both localized families largely swap among identical-valued bytes. Cyclic shift disturbs ~12× more byte values (drags real payload content into padded regions) and is correspondingly the most detectable of the four.

TEST C — CICIDS2017 replication (complete, 5/5 seeds)

C1 — new surrogates (retrained on CICIDS2017, seed 42/5000 samples, saved as surrogate_mlp_cicids.joblib / surrogate_rf_cicids.joblib, not reused from UNSW): MLP acc = 97.7%, RF acc = 99.4%, majority-class baseline = 74.16%. Both well above chance — the run is valid on this criterion.

C2 — S/SG per seed (OPTICS):

┌──────┬───────┬───────┬──────────┐
│ Seed │   S   │  SG   │ Δ (SG−S) │
├──────┼───────┼───────┼──────────┤
│ 42   │ 1.40% │ 1.00% │ −0.40    │
├──────┼───────┼───────┼──────────┤
│ 123  │ 3.40% │ 1.20% │ −2.20    │
├──────┼───────┼───────┼──────────┤
│ 456  │ 4.20% │ 4.60% │ +0.40    │
├──────┼───────┼───────┼──────────┤
│ 789  │ 0.00% │ 4.20% │ +4.20    │
├──────┼───────┼───────┼──────────┤
│ 1024 │ 7.40% │ 2.40% │ −5.00    │
└──────┴───────┴───────┴──────────┘

C3 — guidance effect: S = 3.28% ± 2.53%, SG = 2.68% ± 1.49%, effect = −0.60 ± 3.03, only 2/5 seeds positive, vs. UNSW's +4.68 ± 1.57, 5/5 seeds positive.

C4 — other algorithms: DBSCAN and MeanShift stayed at 0.00% throughout, matching UNSW. But HDBSCAN is nonzero on 8 of 10 CICIDS cells (2.0%–6.8%) — on UNSW, HDBSCAN stayed at ~0% across every variant in this entire investigation. Validity was 100% throughout (by construction, as expected).

Verdict: the guidance effect does NOT replicate on CICIDS2017 — stated plainly. Sign flips per-seed (3 of 5 negative), the mean crosses zero, and the magnitude (−0.60±3.03) bears no resemblance to UNSW's stable +4.68±1.57. This is reported as a genuine negative result, not rescued by re-seeding or re-scoping: the guidance effect is dataset-dependent, not a general property of this attack/surrogate/defense combination. The HDBSCAN divergence is a second, independent sign that CICIDS2017 behaves structurally differently under this pipeline, not just noisier.

Scope paragraph

The permutation/count-invariance result is proof-backed and configuration-independent: it held at both binarization thresholds the source paper is internally ambiguous about, and across four structurally distinct permutation families, with zero exceptions across all 1,000 count checks run in this session. No amount of further data can strengthen or weaken it — only its boundary was being probed, and it did not break. Test B's further empirical finding — that capture varies 0%–6.3% across these families despite identical count-invariance — establishes that detection is mediated by spatial structure specifically, correcting a naive "swap attacks are invisible" reading to "swap attacks are attenuated, through a specific and now-identified mechanism, in a way that itself varies by permutation family." The guidance-effect finding is empirical and its scope is now sharply bounded, in the negative direction: it is supported by a second configuration (threshold=0.3 on UNSW-NB15, same sign, same seed-count) but explicitly refuted as a general, dataset-independent effect by the one dataset replication run — CICIDS2017 shows a near-zero, sign-unstable effect. The honest current scope of the guidance-effect claim is: "positive and stable across binarization thresholds on UNSW-NB15, 5 seeds; does not replicate on CICIDS2017." Any future lens treating +4.68pts as a general property of this substrate should be corrected — it is a UNSW-NB15-specific finding as of this test.

Nothing built, fixed, or written to CLAUDE.md/cc_summary.md. Flagging for your decision on how (or whether) to fold the CICIDS non-replication into the documented Lens 4 record.