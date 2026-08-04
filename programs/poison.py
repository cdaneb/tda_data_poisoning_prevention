"""
Data poisoning module.

Creates poisoned versions of datasets by injecting adversarial examples.
The paper uses sophisticated attack methods (Chale et al. 2023 genetic algorithm,
Hore et al. 2023 deep RL). For our baseline reproduction, we use a simplified
perturbation approach that modifies payload bytes of selected samples.

The poisoning simulates a data injection attack where adversarial samples
are added to the training set. The attacker modifies malicious traffic
to appear benign-like while retaining some anomalous structure.
"""
import numpy as np


def poison_dataset(X, y, poison_rate=0.10, noise_scale=30, random_state=42):
    """
    Create a poisoned version of the dataset.

    Strategy: Select a fraction of samples, perturb their payload bytes
    by adding controlled noise, and mark them as poisoned. This simulates
    the adversarial modification described in the paper where attackers
    modify malicious payloads to evade detection.

    Args:
        X: np.ndarray of shape (N, 1500) — original payload bytes (0-255)
        y: np.ndarray of shape (N,) — original labels
        poison_rate: float — fraction of samples to poison (default 0.10 = 10%)
        noise_scale: int — standard deviation of Gaussian noise to add
        random_state: int — for reproducibility

    Returns:
        X_combined: np.ndarray (N + N_poison, 1500) — original + poisoned samples
        y_combined: np.ndarray (N + N_poison,) — original + poisoned labels
        is_poisoned: np.ndarray (N + N_poison,) bool — True for poisoned samples
    """
    rng = np.random.RandomState(random_state)

    N = len(X)
    n_poison = int(N * poison_rate)

    # Select samples to poison (sample from the full dataset)
    poison_indices = rng.choice(N, size=n_poison, replace=False)
    X_poison = X[poison_indices].copy().astype(np.float64)
    y_poison = y[poison_indices].copy()

    # Apply perturbation: add Gaussian noise, clamp to [0, 255]
    noise = rng.normal(0, noise_scale, size=X_poison.shape)
    X_poison = np.clip(X_poison + noise, 0, 255).astype(np.uint8)

    # Additionally, apply some byte substitutions to simulate
    # the "functionally equivalent code substitution" from Chale et al.
    # Randomly swap some byte values within each poisoned sample
    for i in range(n_poison):
        n_swaps = rng.randint(10, 50)
        swap_positions = rng.choice(1500, size=n_swaps * 2, replace=False)
        for j in range(0, len(swap_positions) - 1, 2):
            pos_a, pos_b = swap_positions[j], swap_positions[j + 1]
            X_poison[i, pos_a], X_poison[i, pos_b] = X_poison[i, pos_b], X_poison[i, pos_a]

    # Combine original and poisoned data
    X_combined = np.vstack([X, X_poison])
    y_combined = np.concatenate([y, y_poison])
    is_poisoned = np.concatenate([
        np.zeros(N, dtype=bool),
        np.ones(n_poison, dtype=bool)
    ])

    print(f"Poisoning complete:")
    print(f"  Original samples: {N}")
    print(f"  Poisoned samples: {n_poison}")
    print(f"  Combined samples: {len(X_combined)}")
    print(f"  Poison rate: {n_poison / len(X_combined):.2%}")

    return X_combined, y_combined, is_poisoned


if __name__ == "__main__":
    # Quick test with synthetic data
    print("=== Testing poison module ===\n")
    X_fake = np.random.randint(0, 256, size=(100, 1500), dtype=np.uint8)
    y_fake = np.array(["normal"] * 50 + ["attack"] * 50)

    X_p, y_p, is_p = poison_dataset(X_fake, y_fake, poison_rate=0.10)

    assert X_p.shape == (110, 1500), f"Expected (110, 1500), got {X_p.shape}"
    assert y_p.shape == (110,), f"Expected (110,), got {y_p.shape}"
    assert is_p.sum() == 10, f"Expected 10 poisoned, got {is_p.sum()}"
    assert X_p.dtype == np.uint8, f"Expected uint8, got {X_p.dtype}"
    assert X_p.min() >= 0 and X_p.max() <= 255

    print("\n=== Poison module verification PASSED ===")
