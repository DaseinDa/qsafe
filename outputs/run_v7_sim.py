"""
Q-SAFE v7 — Qiskit Statevector Simulation Experiment
=======================================================
8-qubit, 3-layer Parametric Quantum Kernel (PQK) circuit with Heisenberg evolution.
Anomaly detection on digits dataset (0-4 normal, 5-9 anomaly).
Compares: PQK-OCSVM, PQK-IF, PQK-Maha, QKernel-OCSVM,
          Classical IF, OCSVM, LOF, Mahalanobis.
"""

import os
import json
import time
import warnings
import numpy as np
import math

warnings.filterwarnings("ignore")

# ── Qiskit imports ──────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector

# ── Scikit-learn imports ────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.covariance import EmpiricalCovariance

# ── Constants ───────────────────────────────────────────────────────────────
N_QUBITS = 8
N_LAYERS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v7_sim")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PQK Circuit — Heisenberg-inspired feature map
# ═══════════════════════════════════════════════════════════════════════════

def build_pqk_circuit(n_qubits: int, n_layers: int) -> tuple[QuantumCircuit, ParameterVector]:
    """Build PQK circuit with Heisenberg evolution layers."""
    params = ParameterVector("x", length=n_qubits)
    qc = QuantumCircuit(n_qubits)

    for layer in range(n_layers):
        # Encoding: Ry rotation for each qubit
        for q in range(n_qubits):
            qc.ry(params[q % len(params)], q)

        # Heisenberg-inspired entanglement: XX+YY+ZZ interactions
        for q in range(n_qubits - 1):
            # ZZ
            qc.cx(q, q + 1)
            qc.rz(np.pi / 4, q + 1)
            qc.cx(q, q + 1)
            # XX
            qc.h(q)
            qc.h(q + 1)
            qc.cx(q, q + 1)
            qc.rz(np.pi / 4, q + 1)
            qc.cx(q, q + 1)
            qc.h(q)
            qc.h(q + 1)
            # YY
            qc.sdg(q)
            qc.sdg(q + 1)
            qc.h(q)
            qc.h(q + 1)
            qc.cx(q, q + 1)
            qc.rz(np.pi / 4, q + 1)
            qc.cx(q, q + 1)
            qc.h(q)
            qc.h(q + 1)
            qc.s(q)
            qc.s(q + 1)

    # Final encoding layer
    for q in range(n_qubits):
        qc.ry(params[q % len(params)], q)

    return qc, params


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Statevector utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_statevector(qc: QuantumCircuit, params: ParameterVector, x: np.ndarray) -> Statevector:
    """Bind parameters and compute statevector."""
    bound = qc.assign_parameters(dict(zip(params, x)))
    return Statevector(bound)


def pauli_expectation(sv: Statevector, n_qubits: int) -> np.ndarray:
    """Compute <Z_i> and <X_i> for each qubit."""
    feats = []
    for q in range(n_qubits):
        # Z expectation
        z_label = "I" * (n_qubits - q - 1) + "Z" + "I" * q
        op = SparsePauliOp(z_label)
        feats.append(sv.expectation_value(op).real)
        # X expectation
        x_label = "I" * (n_qubits - q - 1) + "X" + "I" * q
        op = SparsePauliOp(x_label)
        feats.append(sv.expectation_value(op).real)
    return np.array(feats)


def fidelity(sv1: Statevector, sv2: Statevector) -> float:
    """Quantum fidelity |<psi|phi>|^2."""
    return float(abs(sv1.inner(sv2)) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_pqk_features(X: np.ndarray, qc: QuantumCircuit, params: ParameterVector) -> np.ndarray:
    """Extract PauliZ/X expectation values as feature vectors."""
    print(f"  Extracting PQK features for {len(X)} samples...", flush=True)
    features = []
    for i, x in enumerate(X):
        if i % 50 == 0:
            print(f"    {i}/{len(X)}", end="\r", flush=True)
        sv = get_statevector(qc, params, x)
        features.append(pauli_expectation(sv, N_QUBITS))
    print(f"    {len(X)}/{len(X)} done.    ")
    return np.array(features)


def build_quantum_kernel(X_train: np.ndarray, X_test: np.ndarray,
                         qc: QuantumCircuit, params: ParameterVector) -> np.ndarray:
    """Build fidelity quantum kernel K[i,j] = |<x_i|x_j>|^2 for test vs train."""
    print(f"  Building quantum kernel ({len(X_test)}x{len(X_train)})...", flush=True)
    # Precompute train statevectors
    sv_train = []
    for i, x in enumerate(X_train):
        if i % 20 == 0:
            print(f"    train sv {i}/{len(X_train)}", end="\r", flush=True)
        sv_train.append(get_statevector(qc, params, x))
    print(f"    train sv {len(X_train)}/{len(X_train)} done.    ")

    K = np.zeros((len(X_test), len(X_train)))
    for i, x in enumerate(X_test):
        if i % 20 == 0:
            print(f"    kernel row {i}/{len(X_test)}", end="\r", flush=True)
        sv_i = get_statevector(qc, params, x)
        for j, sv_j in enumerate(sv_train):
            K[i, j] = fidelity(sv_i, sv_j)
    print(f"    kernel row {len(X_test)}/{len(X_test)} done.    ")
    return K


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Anomaly detectors
# ═══════════════════════════════════════════════════════════════════════════

def mahalanobis_scores(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Return Mahalanobis distance as anomaly score (higher = more anomalous)."""
    cov = EmpiricalCovariance().fit(X_train)
    return cov.mahalanobis(X_test)


def qkernel_ocsvm_scores(K_train: np.ndarray, K_test: np.ndarray) -> np.ndarray:
    """One-Class SVM with precomputed quantum kernel."""
    # K_train: (n_train, n_train), K_test: (n_test, n_train)
    clf = OneClassSVM(kernel="precomputed", nu=0.1)
    clf.fit(K_train)
    return -clf.decision_function(K_test)  # higher = more anomalous


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main experiment
# ═══════════════════════════════════════════════════════════════════════════

SEEDS = [42, 7, 13, 99, 2025]
N_TRAIN = 200   # normal samples for training
N_TEST_NORMAL = 100
N_TEST_ANOMALY = 100


def run_seed(seed: int, qc: QuantumCircuit, params: ParameterVector) -> dict:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    rng = np.random.RandomState(seed)

    # ── Load & split digits ──────────────────────────────────────────────
    digits = load_digits()
    X_raw, y_raw = digits.data, digits.target

    normal_idx = np.where(y_raw <= 4)[0]
    anomaly_idx = np.where(y_raw >= 5)[0]

    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    train_idx = normal_idx[:N_TRAIN]
    test_normal_idx = normal_idx[N_TRAIN: N_TRAIN + N_TEST_NORMAL]
    test_anomaly_idx = anomaly_idx[:N_TEST_ANOMALY]

    X_train_raw = X_raw[train_idx]
    X_test_raw = np.concatenate([X_raw[test_normal_idx], X_raw[test_anomaly_idx]])
    y_test = np.array([0] * N_TEST_NORMAL + [1] * N_TEST_ANOMALY)  # 1 = anomaly

    # ── PCA → 8 features, scale to [0, π] ──────────────────────────────
    pca = PCA(n_components=N_QUBITS, random_state=seed)
    scaler = MinMaxScaler(feature_range=(0, math.pi))

    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # ── PQK feature extraction ───────────────────────────────────────────
    print("\n[PQK Feature Extraction]")
    t0 = time.time()
    F_train = extract_pqk_features(X_train_scaled, qc, params)
    F_test = extract_pqk_features(X_test_scaled, qc, params)
    t_feat = time.time() - t0
    print(f"  Feature extraction: {t_feat:.1f}s")

    # ── Quantum Kernel ───────────────────────────────────────────────────
    print("\n[Quantum Kernel]")
    t0 = time.time()
    # Train kernel (for OCSVM fit) — subsample to keep tractable
    K_train_sub_idx = rng.choice(len(X_train_scaled), size=min(80, len(X_train_scaled)), replace=False)
    X_train_k = X_train_scaled[K_train_sub_idx]
    K_train = build_quantum_kernel(X_train_k, X_train_k, qc, params)
    K_test = build_quantum_kernel(X_train_k, X_test_scaled, qc, params)
    t_kernel = time.time() - t0
    print(f"  Kernel build: {t_kernel:.1f}s")

    # ── Classical baselines ──────────────────────────────────────────────
    results = {}

    # 1. PQK-OCSVM
    clf = OneClassSVM(kernel="rbf", nu=0.1)
    clf.fit(F_train)
    scores = -clf.decision_function(F_test)
    results["PQK-OCSVM"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 2. PQK-IF
    clf = IsolationForest(n_estimators=100, random_state=seed)
    clf.fit(F_train)
    scores = -clf.score_samples(F_test)
    results["PQK-IF"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 3. PQK-Maha
    scores = mahalanobis_scores(F_train, F_test)
    results["PQK-Maha"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 4. QKernel-OCSVM
    scores = qkernel_ocsvm_scores(K_train, K_test)
    results["QKernel-OCSVM"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 5. Classical IF (raw PCA features)
    clf = IsolationForest(n_estimators=100, random_state=seed)
    clf.fit(X_train_scaled)
    scores = -clf.score_samples(X_test_scaled)
    results["Classical-IF"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 6. Classical OCSVM
    clf = OneClassSVM(kernel="rbf", nu=0.1)
    clf.fit(X_train_scaled)
    scores = -clf.decision_function(X_test_scaled)
    results["Classical-OCSVM"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 7. LOF
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
    clf.fit(X_train_scaled)
    scores = -clf.score_samples(X_test_scaled)
    results["LOF"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    # 8. Mahalanobis (classical)
    scores = mahalanobis_scores(X_train_scaled, X_test_scaled)
    results["Mahalanobis"] = (roc_auc_score(y_test, scores), average_precision_score(y_test, scores))

    print("\n  Results this seed:")
    print(f"  {'Method':<20} {'ROC-AUC':>8} {'AP':>8}")
    print(f"  {'-'*38}")
    for k, (auc, ap) in results.items():
        print(f"  {k:<20} {auc:>8.4f} {ap:>8.4f}")

    return {
        "seed": seed,
        "t_feat": t_feat,
        "t_kernel": t_kernel,
        "results": {k: {"roc_auc": v[0], "avg_precision": v[1]} for k, v in results.items()}
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\nQ-SAFE v7 — Statevector Simulation")
    print(f"  Qubits: {N_QUBITS}  |  Layers: {N_LAYERS}  |  Seeds: {SEEDS}")
    print(f"  Results → {RESULTS_DIR}\n")

    # Build circuit once
    qc, params = build_pqk_circuit(N_QUBITS, N_LAYERS)
    print(f"Circuit depth: {qc.depth()}  |  Gates: {qc.size()}\n")

    all_seed_results = []
    for seed in SEEDS:
        sr = run_seed(seed, qc, params)
        all_seed_results.append(sr)

    # ── Aggregate ────────────────────────────────────────────────────────
    methods = list(all_seed_results[0]["results"].keys())
    agg = {m: {"roc_auc": [], "avg_precision": []} for m in methods}
    for sr in all_seed_results:
        for m in methods:
            agg[m]["roc_auc"].append(sr["results"][m]["roc_auc"])
            agg[m]["avg_precision"].append(sr["results"][m]["avg_precision"])

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — Q-SAFE v7 (Statevector Sim, 5 seeds)")
    print("=" * 65)
    print(f"  {'Method':<20} {'ROC-AUC':>10} {'±':>6} {'AP':>10} {'±':>6}")
    print(f"  {'-'*56}")
    summary_rows = []
    for m in methods:
        aucs = agg[m]["roc_auc"]
        aps = agg[m]["avg_precision"]
        row = {
            "method": m,
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "ap_mean": float(np.mean(aps)),
            "ap_std": float(np.std(aps)),
        }
        summary_rows.append(row)
        print(f"  {m:<20} {row['roc_auc_mean']:>10.4f} {row['roc_auc_std']:>6.4f} "
              f"{row['ap_mean']:>10.4f} {row['ap_std']:>6.4f}")
    print("=" * 65)

    # ── Save results ─────────────────────────────────────────────────────
    out = {
        "experiment": "Q-SAFE v7 Statevector Simulation",
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "seeds": SEEDS,
        "summary": summary_rows,
        "per_seed": all_seed_results,
    }
    out_path = os.path.join(RESULTS_DIR, "results_v7_sim.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── IBM Quantum connection check ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  IBM Quantum Connection Check")
    print("=" * 65)
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token="YfL6HgP9B4h-gKqqVtivY0cP85ZgUdB6Kr3sdz7qYhKM"
        )
        backends = service.backends()
        print(f"  Connected! Available backends ({len(backends)}):")
        for b in backends:
            print(f"    - {b.name}")
    except Exception as e:
        print(f"  IBM Quantum connection failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
