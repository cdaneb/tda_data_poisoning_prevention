"""
Downstream classifier evaluation.

Trains a simple ML classifier on three versions of the data:
  1. Original clean data (ideal baseline — no poisoning)
  2. Poisoned data (worst case — poisoning present, no defense)
  3. Sanitized data (our method — iterative TDA filtration applied)

Compares classification accuracy, F1 score, and false positive rate.
This demonstrates the practical security impact of our iterative approach.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from data_loader import load_unsw
from poison import poison_dataset
from iterative_filter import iterative_filter


def evaluate_classifier(X_train, y_train, X_test, y_test, label=""):
    """Train a Random Forest and evaluate on test set."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_enc)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average="weighted")

    print(f"\n  [{label}]")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

    return {"accuracy": acc, "f1_weighted": f1, "train_size": len(X_train)}


def run_classifier_experiment(max_samples=5000, poison_rate=0.10):
    """
    Full classifier evaluation comparing clean, poisoned, and sanitized training data.
    """
    print("=" * 70)
    print("DOWNSTREAM CLASSIFIER EVALUATION")
    print("=" * 70)

    # Load data
    X_all, y_all = load_unsw(max_samples=max_samples)

    # Split into train/test BEFORE poisoning (test set stays clean)
    X_train_clean, X_test, y_train_clean, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\nClean train: {len(X_train_clean)}, Test: {len(X_test)}")

    # Create poisoned training set
    X_train_poisoned, y_train_poisoned, is_poisoned = poison_dataset(
        X_train_clean, y_train_clean, poison_rate=poison_rate
    )

    # Run iterative filtration on poisoned training data
    print(f"\nRunning iterative filtration on poisoned training data...")
    filter_result = iterative_filter(
        X_train_poisoned, is_poisoned,
        algorithm="OPTICS",
        max_iterations=10,
        verbose=True
    )

    # Build sanitized training set (sanitized pool + residual, minus poisoned pool)
    safe_indices = np.concatenate([
        filter_result["sanitized_indices"],
        filter_result["residual_indices"]
    ])
    X_train_sanitized = X_train_poisoned[safe_indices]
    y_train_sanitized = y_train_poisoned[safe_indices]

    # Evaluate all three scenarios
    print(f"\n{'='*70}")
    print("CLASSIFIER COMPARISON")
    print(f"{'='*70}")

    r1 = evaluate_classifier(X_train_clean, y_train_clean, X_test, y_test,
                             "Clean Training Data (Ideal)")
    r2 = evaluate_classifier(X_train_poisoned, y_train_poisoned, X_test, y_test,
                             "Poisoned Training Data (No Defense)")
    r3 = evaluate_classifier(X_train_sanitized, y_train_sanitized, X_test, y_test,
                             "Sanitized Training Data (Our Method)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<40} {'Accuracy':>10} {'F1':>10} {'TrainSize':>10}")
    print(f"  {'-'*70}")
    print(f"  {'Clean (ideal baseline)':<40} {r1['accuracy']:>10.4f} {r1['f1_weighted']:>10.4f} {r1['train_size']:>10}")
    print(f"  {'Poisoned (no defense)':<40} {r2['accuracy']:>10.4f} {r2['f1_weighted']:>10.4f} {r2['train_size']:>10}")
    print(f"  {'Sanitized (our method)':<40} {r3['accuracy']:>10.4f} {r3['f1_weighted']:>10.4f} {r3['train_size']:>10}")

    acc_recovery = (r3["accuracy"] - r2["accuracy"]) / (r1["accuracy"] - r2["accuracy"]) * 100 \
        if r1["accuracy"] != r2["accuracy"] else float('nan')
    print(f"\n  Accuracy recovery: {acc_recovery:.1f}% of the gap between poisoned and clean")


if __name__ == "__main__":
    run_classifier_experiment(max_samples=5000, poison_rate=0.10)
