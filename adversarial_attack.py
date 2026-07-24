"""
Lens 4 — Surrogate-guided (Chale-approximation) adversarial poisoning attack.

Drop-in alternative to poison.poison_dataset() with the identical
(X_combined, y_combined, is_poisoned) contract, so run_baseline.py /
run_iterative.py / run_multi_seed.py / classifier_eval.py can swap it in
without any change to their own code (Phase 3 wiring; not done here).

Two modes:
  - "random_swap": poison.py's existing attack, unchanged, reused verbatim
    (delegates to poison_dataset) so the equivalence gate holds by
    construction rather than by re-implementation.
  - "chale_ga": a surrogate-loss-guided generational search over the same
    byte-swap move family, approximating Chale et al. (2023)'s
    functionally-equivalent substitution attack. Targets the malicious
    class only (a documented divergence from poison.py's label-agnostic
    random selection) and maximizes the surrogate's benign-class
    probability for each targeted sample.

poison.py is not modified; it remains the reference Gate A compares against.
"""
import time
import numpy as np
from poison import poison_dataset


# ---------------------------------------------------------------------------
# Mode: random_swap (reference / legacy — delegates to poison.py)
# ---------------------------------------------------------------------------

def random_swap_attack(X, y, poison_rate=0.10, noise_scale=30, random_state=42):
    """Replicates poison.py's current behavior exactly, by delegation."""
    return poison_dataset(X, y, poison_rate=poison_rate, noise_scale=noise_scale,
                           random_state=random_state)


# ---------------------------------------------------------------------------
# Mode: R60 (magnitude control) — malicious-only targeting, matched swap
# count to chale_ga, but a single undirected random draw (no surrogate, no
# selection). Isolates "more perturbation + malicious targeting" from
# "adversarial guidance" (the R60 -> G60 contrast Phase 3 needs).
# ---------------------------------------------------------------------------

def malicious_random_attack(X, y, poison_rate=0.10, random_state=42, n_swaps=60):
    """R60: malicious-only targets, one random n_swaps-swap set each, no guidance."""
    rng = np.random.default_rng(random_state)
    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]

    N = len(X)
    n_poison = int(N * poison_rate)
    if n_poison > len(malicious_idx):
        raise ValueError(f"poison_rate implies {n_poison} targets but only "
                          f"{len(malicious_idx)} malicious samples exist")

    target_indices = rng.choice(malicious_idx, size=n_poison, replace=False)

    X_poison = np.zeros((n_poison, X.shape[1]), dtype=np.uint8)
    y_poison = y[target_indices].copy()
    attack_log = []
    for i, idx in enumerate(target_indices):
        sample = X[idx]
        swap_set = _random_swap_set(n_swaps, sample.shape[0], rng)
        perturbed = _apply_swap_set(sample, swap_set)
        X_poison[i] = perturbed
        valid = (perturbed.min() >= 0 and perturbed.max() <= 255 and
                 sorted(perturbed.tolist()) == sorted(sample.tolist()))
        attack_log.append({"target_index": int(idx), "valid": bool(valid)})

    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([np.zeros(N, dtype=bool), np.ones(n_poison, dtype=bool)])
    return X_combined, y_combined, is_poisoned, attack_log


# ---------------------------------------------------------------------------
# Test B — other multiset-preserving permutation families (Phase P). Each
# mirrors malicious_random_attack's shape exactly (malicious-only targeting,
# one random draw per sample, no guidance) so the only variable between
# families is the permutation itself. Original definitions from the prior
# Test B run are not recoverable from this repo, so each convention below is
# stated explicitly; a reproduction gap should be checked against these
# conventions before being treated as a substantive discrepancy.
# ---------------------------------------------------------------------------

def block_reversal_attack(X, y, poison_rate=0.10, random_state=42, k=120):
    """
    Reverse one contiguous k-byte block at a random offset.

    Convention: offset ~ Uniform{0, ..., n_bytes - k} (no wraparound — the
    block never wraps past position n_bytes-1). Bytes outside the block are
    untouched. A reversal is its own inverse and trivially preserves the
    byte multiset (it's a permutation of positions within the block).
    """
    rng = np.random.default_rng(random_state)
    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]

    N = len(X)
    n_bytes = X.shape[1]
    n_poison = int(N * poison_rate)
    if n_poison > len(malicious_idx):
        raise ValueError(f"poison_rate implies {n_poison} targets but only "
                          f"{len(malicious_idx)} malicious samples exist")
    if k > n_bytes:
        raise ValueError(f"k={k} exceeds n_bytes={n_bytes}")

    target_indices = rng.choice(malicious_idx, size=n_poison, replace=False)

    X_poison = np.zeros((n_poison, n_bytes), dtype=np.uint8)
    y_poison = y[target_indices].copy()
    attack_log = []
    for i, idx in enumerate(target_indices):
        sample = X[idx]
        offset = int(rng.integers(0, n_bytes - k + 1))
        perturbed = sample.copy()
        perturbed[offset:offset + k] = perturbed[offset:offset + k][::-1]
        X_poison[i] = perturbed
        valid = (perturbed.min() >= 0 and perturbed.max() <= 255 and
                 sorted(perturbed.tolist()) == sorted(sample.tolist()))
        attack_log.append({"target_index": int(idx), "offset": offset, "valid": bool(valid)})

    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([np.zeros(N, dtype=bool), np.ones(n_poison, dtype=bool)])
    return X_combined, y_combined, is_poisoned, attack_log


def block_swap_attack(X, y, poison_rate=0.10, random_state=42, k=60):
    """
    Exchange two disjoint contiguous k-byte blocks.

    Convention: two offsets are drawn i.i.d. Uniform{0, ..., n_bytes - k}
    and resampled (rejection sampling) until the two k-byte blocks don't
    overlap (|offset_a - offset_b| >= k). The two blocks are exchanged
    whole — order within each block is preserved, only their positions swap
    ("2 x k" bytes moved total, k per block). Rejection sampling is cheap
    here: n_bytes=1500 is large relative to k=60, so collisions are rare.
    """
    rng = np.random.default_rng(random_state)
    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]

    N = len(X)
    n_bytes = X.shape[1]
    n_poison = int(N * poison_rate)
    if n_poison > len(malicious_idx):
        raise ValueError(f"poison_rate implies {n_poison} targets but only "
                          f"{len(malicious_idx)} malicious samples exist")
    if 2 * k > n_bytes:
        raise ValueError(f"2*k={2*k} exceeds n_bytes={n_bytes}; blocks cannot be disjoint")

    target_indices = rng.choice(malicious_idx, size=n_poison, replace=False)

    X_poison = np.zeros((n_poison, n_bytes), dtype=np.uint8)
    y_poison = y[target_indices].copy()
    attack_log = []
    for i, idx in enumerate(target_indices):
        sample = X[idx]
        while True:
            offset_a = int(rng.integers(0, n_bytes - k + 1))
            offset_b = int(rng.integers(0, n_bytes - k + 1))
            if abs(offset_a - offset_b) >= k:
                break
        perturbed = sample.copy()
        block_a = sample[offset_a:offset_a + k].copy()
        block_b = sample[offset_b:offset_b + k].copy()
        perturbed[offset_a:offset_a + k] = block_b
        perturbed[offset_b:offset_b + k] = block_a
        X_poison[i] = perturbed
        valid = (perturbed.min() >= 0 and perturbed.max() <= 255 and
                 sorted(perturbed.tolist()) == sorted(sample.tolist()))
        attack_log.append({"target_index": int(idx), "offset_a": offset_a,
                            "offset_b": offset_b, "valid": bool(valid)})

    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([np.zeros(N, dtype=bool), np.ones(n_poison, dtype=bool)])
    return X_combined, y_combined, is_poisoned, attack_log


def cyclic_shift_attack(X, y, poison_rate=0.10, random_state=42):
    """
    Rotate the full byte vector by a random offset.

    Convention: shift ~ Uniform{1, ..., n_bytes - 1} (excludes 0 and
    n_bytes, both of which are the identity permutation under rotation).
    Implemented via np.roll, which wraps rather than truncating.
    """
    rng = np.random.default_rng(random_state)
    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]

    N = len(X)
    n_bytes = X.shape[1]
    n_poison = int(N * poison_rate)
    if n_poison > len(malicious_idx):
        raise ValueError(f"poison_rate implies {n_poison} targets but only "
                          f"{len(malicious_idx)} malicious samples exist")

    target_indices = rng.choice(malicious_idx, size=n_poison, replace=False)

    X_poison = np.zeros((n_poison, n_bytes), dtype=np.uint8)
    y_poison = y[target_indices].copy()
    attack_log = []
    for i, idx in enumerate(target_indices):
        sample = X[idx]
        shift = int(rng.integers(1, n_bytes))
        perturbed = np.roll(sample, shift)
        X_poison[i] = perturbed
        valid = (perturbed.min() >= 0 and perturbed.max() <= 255 and
                 sorted(perturbed.tolist()) == sorted(sample.tolist()))
        attack_log.append({"target_index": int(idx), "shift": shift, "valid": bool(valid)})

    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([np.zeros(N, dtype=bool), np.ones(n_poison, dtype=bool)])
    return X_combined, y_combined, is_poisoned, attack_log


# ---------------------------------------------------------------------------
# Surrogate labeling / training
# ---------------------------------------------------------------------------

def label_to_binary(y):
    """UNSW-NB15 category strings -> benign(0)/malicious(1). 'normal' -> 0."""
    return np.array([0 if str(label).lower() == "normal" else 1 for label in y], dtype=int)


def train_surrogates(X, y, random_state=42, test_size=0.2, verbose=True):
    """
    Train the primary (MLPClassifier) and secondary (RandomForestClassifier)
    surrogates on raw payload bytes, scaled to [0, 1].

    Returns a dict with the fitted models, the scaling function, and held-out
    accuracy for each (Gate B1).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    y_bin = label_to_binary(y)
    X_scaled = X.astype(np.float64) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )

    mlp = MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=300,
                         random_state=random_state, early_stopping=True)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    if verbose:
        print(f"  Surrogate held-out accuracy: MLP={mlp_acc:.4f}, RF={rf_acc:.4f} "
              f"(n_test={len(y_test)}, class balance benign/malicious in test: "
              f"{(y_test == 0).sum()}/{(y_test == 1).sum()})")

    return {
        "mlp": mlp, "rf": rf,
        "mlp_acc": mlp_acc, "rf_acc": rf_acc,
        "scale_fn": lambda Xraw: Xraw.astype(np.float64) / 255.0,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    }


def benign_proba(model, X_scaled_row):
    """P(benign) for a single scaled sample, as a 2D-input model expects."""
    return model.predict_proba(X_scaled_row.reshape(1, -1))[0, 0]


# ---------------------------------------------------------------------------
# Chale-approximation GA search over byte-swap sets
# ---------------------------------------------------------------------------

def _random_swap_set(n_swaps, n_bytes, rng):
    """A candidate move: n_swaps disjoint (pos_a, pos_b) byte-position pairs."""
    positions = rng.choice(n_bytes, size=n_swaps * 2, replace=False)
    return [(int(positions[j]), int(positions[j + 1])) for j in range(0, len(positions) - 1, 2)]


def _mutate_swap_set(swap_set, n_swaps, n_bytes, rng, mutation_rate=0.3):
    """Perturb a swap set: with mutation_rate, replace each pair with a fresh random one."""
    new_set = []
    for pair in swap_set:
        if rng.random() < mutation_rate:
            positions = rng.choice(n_bytes, size=2, replace=False)
            new_set.append((int(positions[0]), int(positions[1])))
        else:
            new_set.append(pair)
    return new_set


def _apply_swap_set(sample, swap_set):
    """Return a copy of `sample` with all swaps in swap_set applied."""
    out = sample.copy()
    for pos_a, pos_b in swap_set:
        out[pos_a], out[pos_b] = out[pos_b], out[pos_a]
    return out


def ga_search_one_sample(sample, model, scale_fn, n_swaps, population_size, n_generations,
                          rng, early_stop_benign_proba=0.5, log_curve=False):
    """
    Generational search over byte-swap sets maximizing the surrogate's
    benign-class probability for one sample. Returns (best_sample,
    best_benign_proba, n_generations_used, [curve if log_curve]).
    """
    n_bytes = sample.shape[0]
    population = [_random_swap_set(n_swaps, n_bytes, rng) for _ in range(population_size)]

    def score_population(pop):
        """Batch-score the whole generation in one predict_proba call.
        Per-call classifier overhead (especially a 100-tree RF's ensemble
        dispatch, ~70x an MLP's per-call cost at batch size 1) is largely
        fixed, not per-sample, so batching amortizes it across the
        population instead of paying it population_size times."""
        perturbed_batch = np.stack([_apply_swap_set(sample, s) for s in pop])
        proba = model.predict_proba(scale_fn(perturbed_batch))[:, 0]
        return proba

    best_swap_set, best_score = None, -1.0
    curve = []
    for gen in range(n_generations):
        scores = score_population(population)
        scored = sorted(zip(population, scores), key=lambda t: t[1], reverse=True)
        if scored[0][1] > best_score:
            best_swap_set, best_score = scored[0]
        if log_curve:
            curve.append(best_score)
        if best_score >= early_stop_benign_proba:
            break
        # Selection: keep top half, mutate to refill
        survivors = [s for s, _ in scored[:max(2, population_size // 2)]]
        next_gen = list(survivors)
        while len(next_gen) < population_size:
            parent = survivors[rng.integers(0, len(survivors))]
            next_gen.append(_mutate_swap_set(parent, n_swaps, n_bytes, rng))
        population = next_gen

    best_sample = _apply_swap_set(sample, best_swap_set)
    result = (best_sample, best_score, gen + 1)
    if log_curve:
        result = result + (curve,)
    return result


def chale_ga_attack(X, y, surrogate_model, scale_fn, poison_rate=0.10, random_state=42,
                     n_swaps=60, population_size=50, n_generations=100,
                     early_stop_benign_proba=0.5, verbose=True):
    """
    Surrogate-guided byte-swap attack (Chale-approximation).

    Target selection: n_poison samples drawn from the MALICIOUS class only
    (a deliberate divergence from poison.py's label-agnostic random draw —
    Chale/Hore are malicious-to-benign evasion attacks, so only malicious
    samples are meaningful attack targets).

    Returns (X_combined, y_combined, is_poisoned, attack_log) — same triple
    contract as poison_dataset, plus a log of per-sample attack outcomes.
    """
    rng = np.random.default_rng(random_state)

    y_bin = label_to_binary(y)
    malicious_idx = np.where(y_bin == 1)[0]

    N = len(X)
    n_poison = int(N * poison_rate)
    if n_poison > len(malicious_idx):
        raise ValueError(f"poison_rate implies {n_poison} targets but only "
                          f"{len(malicious_idx)} malicious samples exist")

    target_indices = rng.choice(malicious_idx, size=n_poison, replace=False)

    X_poison = np.zeros((n_poison, X.shape[1]), dtype=np.uint8)
    y_poison = y[target_indices].copy()
    attack_log = []

    t0 = time.time()
    for i, idx in enumerate(target_indices):
        sample = X[idx]
        best_sample, best_proba, gens_used = ga_search_one_sample(
            sample, surrogate_model, scale_fn, n_swaps, population_size, n_generations,
            rng, early_stop_benign_proba=early_stop_benign_proba,
        )
        X_poison[i] = best_sample
        flipped = best_proba >= 0.5
        # Validity: swap-based moves preserve the byte multiset and range by
        # construction; verify rather than assume.
        valid = (best_sample.min() >= 0 and best_sample.max() <= 255 and
                 sorted(best_sample.tolist()) == sorted(sample.tolist()))
        attack_log.append({
            "target_index": int(idx), "final_benign_proba": float(best_proba),
            "flipped_to_benign": bool(flipped), "generations_used": int(gens_used),
            "valid": bool(valid),
        })
        if verbose and (i + 1) % max(1, n_poison // 10) == 0:
            print(f"    chale_ga: {i+1}/{n_poison} targets processed "
                  f"({time.time()-t0:.1f}s elapsed)")

    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([np.zeros(N, dtype=bool), np.ones(n_poison, dtype=bool)])

    if verbose:
        n_valid = sum(l["valid"] for l in attack_log)
        n_flipped = sum(l["flipped_to_benign"] for l in attack_log)
        print(f"  chale_ga complete: {n_poison} targets, {time.time()-t0:.1f}s total")
        print(f"    Validity: {n_valid}/{n_poison} ({n_valid/n_poison*100:.1f}%)")
        print(f"    Flipped to benign (surrogate): {n_flipped}/{n_poison} ({n_flipped/n_poison*100:.1f}%)")

    return X_combined, y_combined, is_poisoned, attack_log


# ---------------------------------------------------------------------------
# Unified entry point matching poison_dataset's contract
# ---------------------------------------------------------------------------

def adversarial_poison_dataset(X, y, poison_rate=0.10, random_state=42, mode="random_swap", **kwargs):
    """
    Unified entry point. mode="random_swap" matches poison_dataset exactly
    (Gate A). mode="chale_ga" requires surrogate_model= and scale_fn= kwargs
    (see chale_ga_attack) and returns a 4-tuple (X_combined, y_combined,
    is_poisoned, attack_log) instead of the 3-tuple, since the log is
    additive information, not a contract-breaking change for callers that
    only unpack the first three.
    """
    if mode == "random_swap":
        return random_swap_attack(X, y, poison_rate=poison_rate, random_state=random_state)
    elif mode == "chale_ga":
        return chale_ga_attack(X, y, poison_rate=poison_rate, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
