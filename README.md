# Q-SAFE — Quantum Anomaly Detection Experiment

Quantum-classical hybrid anomaly detection using Parametric Quantum Kernels (PQK).

## Structure

```
outputs/          # Experiment scripts
results_v7_sim/   # v7 Statevector simulation results
```

## v7 Experiment

- 8-qubit, 3-layer PQK circuit with Heisenberg evolution
- Dataset: sklearn digits (0-4 normal, 5-9 anomaly), PCA → 8 features, scaled to [0, π]
- Methods: PQK-OCSVM, PQK-IF, PQK-Maha, QKernel-OCSVM, Classical-IF, OCSVM, LOF, Mahalanobis
- 5 seeds

## Requirements

```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime scikit-learn numpy
```

## Run

```bash
python outputs/run_v7_sim.py
```
