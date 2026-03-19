# 15-Qubit Decoherence-Reversal Protocol
### Benchmarking Mid-Circuit Measurements and Environmental Imprinting

**Dwight Kahng** — Independent Researcher  
**Paper:** *A 15-Qubit Decoherence-Reversal Protocol for Benchmarking Mid-Circuit Measurements and Environmental Imprinting* (March 18, 2026)  
**arXiv:** [link to be added]  
**Zenodo:** [link to be added]

---

## Overview

This repository contains the full implementation of a 15-qubit Loschmidt-echo protocol designed to benchmark mid-circuit measurement (MCM) performance on near-term quantum hardware. The protocol is intentionally restricted to a state-vector tractable regime (15 qubits), providing an exactly computable ideal baseline (F_max) against which noisy hardware recovery can be directly compared.

The protocol consists of three stages:
1. **Forward scrambling** — a 15-cycle unitary U applied to a 6-qubit system entangled with a 5-qubit artificial environment
2. **Mid-circuit QND probe measurement** — 4 probe qubits measure coarse-grained system observables as a proxy for environmental imprinting
3. **Loschmidt-echo reversal** — exact logical inverse U† is applied; recovery fidelity on the 11-qubit target register is measured

A surrogate noise model calibrated to published Quantinuum H2-1 specifications provides expected recovery thresholds. A strict 2σ decision rule classifies hardware results as Consistent, Anomalous, or Inconclusive.

---

## Repository Contents

| File | Description |
|------|-------------|
| `local_simulation_v48_master.py` | Local Aer simulation — computes F_max, runs Table 2 sensitivity sweep, and reproduces the Phase VI anomalous decoherence classification using the observed emulator result |
| `azure_quantinuum_submission_v48_public.py` | Azure Quantum submission — connects to Quantinuum H2-1E emulator, runs sweep, submits 400-shot production job via bare-metal OpenQASM 2.0 bypass, classifies result |

### Note on Register Structure

The two scripts use different classical register layouts by design:

- **Local simulation** uses three named registers (`cp`, `cs`, `ce`) — readable, Qiskit-native, compatible with Aer
- **Azure submission** uses a single flat 15-bit register — required for deterministic bitstring concatenation in the Quantinuum JSON payload

Both scripts implement identical quantum physics. The register difference is purely a classical infrastructure adaptation. See Section 3.1 of the paper for details.

---

## Key Results (v48)

| Metric | Value |
|--------|-------|
| F_max (exact ideal baseline) | 0.9889 |
| F_expected (H2 Typical surrogate) | 0.8353 ± 0.0054 |
| F_recovered (400-shot emulator) | 0.7836 ± 0.0212 |
| Table 3 classification | Anomalous Decoherence* |

*The "Anomalous Decoherence" classification reflects a statistically significant gap between the idealized local surrogate model and the full-stack emulator, consistent with unmodeled compilation/routing overhead. It does not indicate unexpected decoherence on physical hardware. This is a successful demonstration of the protocol's diagnostic sensitivity to model mismatch.

---

## Surrogate Calibration (Table 2)

Noise profiles calibrated to published Quantinuum H2-1 specifications:

| Scenario | F_clean ± SE | F_exp ± SE | δ |
|----------|-------------|-----------|---|
| H2 Typical Profile | 0.9884 ± 0.0014 | 0.8353 ± 0.0054 | 0.1531 |
| H2 Pessimistic Limit | 0.9880 ± 0.0013 | 0.6805 ± 0.0071 | 0.3075 |
| Deep Scramble (H2) | 0.9902 ± 0.0013 | 0.7831 ± 0.0064 | 0.2071 |
| Strong QND Kick (H2) | 0.9556 ± 0.0029 | 0.8058 ± 0.0055 | 0.1498 |

---

## Dependencies

```bash
pip install qiskit qiskit-aer
```

For Azure submission only:
```bash
pip install azure-quantum[qiskit] azure-identity
```

---

## Running Locally

```bash
python local_simulation_v48_master.py
```

No Azure account required. Runs full F_max computation, Table 2 sweep on local Aer simulator, and a Phase VI diagnostic test reproducing the anomalous decoherence classification from the observed emulator result. Expect approximately 30 minutes on a standard Colab instance.

---

## Running on Azure Quantum (Quantinuum Emulator)

```bash
export AZURE_QUANTUM_RESOURCE_ID="<your-resource-id>"
export AZURE_QUANTUM_TENANT_ID="<your-tenant-id>"
python azure_quantinuum_submission_v48_public.py
```

You will be prompted to authenticate via browser (device code flow).

### Important — Azure QIR Bypass

Standard `backend.run()` via the Azure QIR compiler was found to mishandle multi-shot mid-circuit measurements on the Quantinuum backend. This script bypasses the QIR compiler by:

1. Compiling the circuit locally to OpenQASM 2.0 via `qasm2.dumps()`
2. Submitting via the bare-metal `Job.from_input_data()` API with Honeywell format tags
3. Injecting `input_params={"count": hardware_shots}` directly into the raw payload

In this project, the bare-metal submission path was required to obtain correct multi-shot behavior on the Quantinuum backend.

**Estimated cost:** ~221 eHQC for a 400-shot run on `quantinuum.sim.h2-1e` (5 eHQC base fee + ~0.54 eHQC/shot).

---

## Citation

If you use this protocol or code, please cite:

```
Kahng, D. (2026). A 15-Qubit Decoherence-Reversal Protocol for Benchmarking
Mid-Circuit Measurements and Environmental Imprinting.
arXiv: [to be added]
```

---

## License

MIT License. See LICENSE file for details.

---

*AI Use Disclosure: The associated manuscript incorporates structural and drafting assistance from large language models. The author remains fully responsible for all content, theoretical claims, and scientific accuracy.*
