"""
Q-SAFE v8 — IBM Torino Hardware Experiment (16-qubit)
======================================================
16-qubit, 2-layer PQK circuit with Heisenberg evolution on ibm_torino.
Uses Qiskit Runtime EstimatorV2 to extract PauliZ/X expectation features.
Compares results with v7 statevector simulation (8-qubit).

Dataset: digits 0-4 (normal), 5-9 (anomaly)
Small scale: 50 train / 30+30 test, 2 seeds — to keep queue time manageable.
"""

import os
import json
import time
import warnings
import numpy as np
import math

warnings.filterwarnings("ignore")

# ── Qiskit ──────────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ── Qiskit Runtime ───────────────────────────────────────────────────────────
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import EstimatorOptions

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
N_QUBITS   = 16
N_LAYERS   = 2          # shallower than v7 to limit noise accumulation
BACKEND    = "ibm_torino"
TOKEN      = "YfL6HgP9B4h-gKqqVtivY0cP85ZgUdB6Kr3sdz7qYhKM"
SEEDS      = [42, 7]
N_TRAIN    = 50
N_TEST_NOR = 30
N_TEST_ANO = 30
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v8_torino")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Circuit
# ═══════════════════════════════════════════════════════════════════════════

def build_pqk_circuit(n_qubits: int, n_layers: int):
    """16-qubit PQK with Heisenberg-inspired nearest-neighbour entanglement."""
    params = ParameterVector("x", length=n_qubits)
    qc = QuantumCircuit(n_qubits)

    for _ in range(n_layers):
        # Encoding
        for q in range(n_qubits):
            qc.ry(params[q], q)

        # Heisenberg ZZ / XX / YY on linear chain (transpiler maps to hardware)
        for q in range(n_qubits - 1):
            # ZZ
            qc.cx(q, q + 1); qc.rz(math.pi / 4, q + 1); qc.cx(q, q + 1)
            # XX
            qc.h(q);  qc.h(q + 1)
            qc.cx(q, q + 1); qc.rz(math.pi / 4, q + 1); qc.cx(q, q + 1)
            qc.h(q);  qc.h(q + 1)
            # YY
            qc.sdg(q); qc.sdg(q + 1); qc.h(q); qc.h(q + 1)
            qc.cx(q, q + 1); qc.rz(math.pi / 4, q + 1); qc.cx(q, q + 1)
            qc.h(q);  qc.h(q + 1); qc.s(q); qc.s(q + 1)

    # Final encoding
    for q in range(n_qubits):
        qc.ry(params[q], q)

    return qc, params


def build_observables(n_qubits: int, layout=None, total_qubits: int = None) -> list:
    """Z_q and X_q for every qubit — 2*n_qubits operators total.
    If layout is provided, apply_layout maps logical → physical qubit space."""
    ops = []
    for q in range(n_qubits):
        z_op = SparsePauliOp("I" * (n_qubits - q - 1) + "Z" + "I" * q)
        x_op = SparsePauliOp("I" * (n_qubits - q - 1) + "X" + "I" * q)
        if layout is not None:
            z_op = z_op.apply_layout(layout, num_qubits=total_qubits)
            x_op = x_op.apply_layout(layout, num_qubits=total_qubits)
        ops.append(z_op)
        ops.append(x_op)
    return ops


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Hardware feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_hardware(X: np.ndarray,
                               qc_t,           # transpiled circuit
                               observables: list,
                               estimator: Estimator,
                               label: str = "") -> tuple[np.ndarray, str]:
    """
    Submit one Estimator job with one PUB per sample.
    Returns (features array, job_id).
    """
    print(f"  Submitting {label} ({len(X)} samples, {len(observables)} observables each)...")
    pubs = [(qc_t, observables, x.tolist()) for x in X]

    job = estimator.run(pubs)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}  — waiting for results (this may take several minutes)...")

    t0 = time.time()
    result = job.result()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # result[i].data.evs has shape (n_observables,)
    features = np.array([result[i].data.evs for i in range(len(X))])
    return features, job_id


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Anomaly detectors
# ═══════════════════════════════════════════════════════════════════════════

def mahalanobis_scores(X_train, X_test):
    return EmpiricalCovariance().fit(X_train).mahalanobis(X_test)

def run_detectors(F_train, F_test, y_test, seed):
    res = {}

    clf = OneClassSVM(kernel="rbf", nu=0.1)
    clf.fit(F_train)
    res["PQK-OCSVM"] = evaluate(-clf.decision_function(F_test), y_test)

    clf = IsolationForest(n_estimators=100, random_state=seed)
    clf.fit(F_train)
    res["PQK-IF"] = evaluate(-clf.score_samples(F_test), y_test)

    res["PQK-Maha"] = evaluate(mahalanobis_scores(F_train, F_test), y_test)

    clf = LocalOutlierFactor(n_neighbors=10, novelty=True)
    clf.fit(F_train)
    res["PQK-LOF"] = evaluate(-clf.score_samples(F_test), y_test)

    return res

def evaluate(scores, y_test):
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "avg_precision": float(average_precision_score(y_test, scores)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Per-seed experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_seed(seed, qc_t, observables, estimator):
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    rng = np.random.RandomState(seed)

    # Data
    digits = load_digits()
    X_raw, y_raw = digits.data, digits.target
    normal_idx  = np.where(y_raw <= 4)[0]
    anomaly_idx = np.where(y_raw >= 5)[0]
    rng.shuffle(normal_idx); rng.shuffle(anomaly_idx)

    X_train_raw  = X_raw[normal_idx[:N_TRAIN]]
    X_test_raw   = np.concatenate([
        X_raw[normal_idx[N_TRAIN: N_TRAIN + N_TEST_NOR]],
        X_raw[anomaly_idx[:N_TEST_ANO]]
    ])
    y_test = np.array([0] * N_TEST_NOR + [1] * N_TEST_ANO)

    # PCA → 16 features, scale to [0, π]
    pca    = PCA(n_components=N_QUBITS, random_state=seed)
    scaler = MinMaxScaler(feature_range=(0, math.pi))
    X_train = scaler.fit_transform(pca.fit_transform(X_train_raw))
    X_test  = scaler.transform(pca.transform(X_test_raw))

    # Hardware feature extraction
    F_train, jid_train = extract_features_hardware(
        X_train, qc_t, observables, estimator, label="TRAIN")
    F_test, jid_test = extract_features_hardware(
        X_test,  qc_t, observables, estimator, label="TEST")

    # Anomaly detection
    results = run_detectors(F_train, F_test, y_test, seed)

    print(f"\n  Results (seed={seed}):")
    print(f"  {'Method':<14} {'ROC-AUC':>8} {'AP':>8}")
    print(f"  {'-'*32}")
    for k, v in results.items():
        print(f"  {k:<14} {v['roc_auc']:>8.4f} {v['avg_precision']:>8.4f}")

    return {
        "seed": seed,
        "job_ids": {"train": jid_train, "test": jid_test},
        "results": results,
        "F_train_shape": list(F_train.shape),
        "F_test_shape": list(F_test.shape),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\nQ-SAFE v8 — IBM Torino Hardware (16-qubit, 2-layer)")
    print(f"  Backend: {BACKEND}  |  Seeds: {SEEDS}")
    print(f"  Train: {N_TRAIN}  |  Test: {N_TEST_NOR}+{N_TEST_ANO}\n")

    # ── Connect ──────────────────────────────────────────────────────────
    print("[1] Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
    backend = service.backend(BACKEND)
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits, {backend.processor_type})")

    # ── Build & transpile circuit ────────────────────────────────────────
    print("\n[2] Building and transpiling circuit...")
    qc, params = build_pqk_circuit(N_QUBITS, N_LAYERS)
    print(f"  Original  — depth: {qc.depth()}  gates: {qc.size()}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    qc_t = pm.run(qc)
    print(f"  Transpiled— depth: {qc_t.depth()}  gates: {qc_t.size()}")

    observables = build_observables(N_QUBITS, layout=qc_t.layout, total_qubits=backend.num_qubits)
    print(f"  Observables: {len(observables)} (Z+X per qubit, mapped to {backend.num_qubits}-qubit space)")

    # ── Estimator with error mitigation ──────────────────────────────────
    print("\n[3] Setting up Estimator (resilience_level=1)...")
    options = EstimatorOptions()
    options.resilience_level = 1          # readout error mitigation + DD
    estimator = Estimator(mode=backend, options=options)

    # ── Run seeds ────────────────────────────────────────────────────────
    print("\n[4] Running experiments...")
    all_results = []
    for seed in SEEDS:
        sr = run_seed(seed, qc_t, observables, estimator)
        all_results.append(sr)
        # Save intermediate results after each seed
        _save(all_results, qc, qc_t)

    # ── Aggregate ────────────────────────────────────────────────────────
    methods = list(all_results[0]["results"].keys())
    agg = {m: {"roc_auc": [], "avg_precision": []} for m in methods}
    for sr in all_results:
        for m in methods:
            agg[m]["roc_auc"].append(sr["results"][m]["roc_auc"])
            agg[m]["avg_precision"].append(sr["results"][m]["avg_precision"])

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — Q-SAFE v8 (ibm_torino, 16-qubit)")
    print("=" * 65)
    print(f"  {'Method':<14} {'ROC-AUC':>10} {'±':>6} {'AP':>10} {'±':>6}")
    print(f"  {'-'*50}")
    summary = []
    for m in methods:
        aucs = agg[m]["roc_auc"]
        aps  = agg[m]["avg_precision"]
        row  = dict(method=m,
                    roc_auc_mean=float(np.mean(aucs)),
                    roc_auc_std=float(np.std(aucs)),
                    ap_mean=float(np.mean(aps)),
                    ap_std=float(np.std(aps)))
        summary.append(row)
        print(f"  {m:<14} {row['roc_auc_mean']:>10.4f} {row['roc_auc_std']:>6.4f}"
              f" {row['ap_mean']:>10.4f} {row['ap_std']:>6.4f}")
    print("=" * 65)

    # ── Compare with v7 simulation ────────────────────────────────────────
    v7_path = os.path.join(os.path.dirname(__file__), "..", "results_v7_sim", "results_v7_sim.json")
    if os.path.exists(v7_path):
        with open(v7_path) as f:
            v7 = json.load(f)
        print("\n  Comparison vs v7 Statevector Sim (8-qubit, 5-seed):")
        print(f"  {'Method':<14} {'v8 HW AUC':>12} {'v7 Sim AUC':>12} {'Delta':>8}")
        print(f"  {'-'*50}")
        v7_map = {r["method"]: r["roc_auc_mean"] for r in v7["summary"]}
        for row in summary:
            v7_auc = v7_map.get(row["method"].replace("PQK-LOF", "LOF"), None)
            delta  = f"{row['roc_auc_mean'] - v7_auc:+.4f}" if v7_auc else "  n/a"
            v7_str = f"{v7_auc:.4f}" if v7_auc else "   n/a"
            print(f"  {row['method']:<14} {row['roc_auc_mean']:>12.4f} {v7_str:>12} {delta:>8}")

    # ── Save final ────────────────────────────────────────────────────────
    _save(all_results, qc, qc_t, summary=summary)
    print(f"\n  Results saved → {RESULTS_DIR}/")
    print("\nDone.")


def _save(all_results, qc, qc_t, summary=None):
    out = {
        "experiment": "Q-SAFE v8 IBM Torino Hardware",
        "backend": BACKEND,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "circuit_depth_original": qc.depth(),
        "circuit_depth_transpiled": qc_t.depth(),
        "seeds": SEEDS,
        "n_train": N_TRAIN,
        "n_test": N_TEST_NOR + N_TEST_ANO,
        "summary": summary,
        "per_seed": all_results,
    }
    with open(os.path.join(RESULTS_DIR, "results_v8_torino.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
