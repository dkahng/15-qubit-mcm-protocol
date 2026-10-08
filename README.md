# 15-Qubit Exactly Simulable Loschmidt-Echo Benchmark
### Mid-Circuit Measurements and Noise-Model Validation

**Dwight Kahng** — Independent Researcher  
**Paper:** *An Exactly Simulable 15-Qubit Loschmidt-Echo Benchmark for Mid-Circuit Measurements and Noise-Model Validation* (v50, October 7, 2026). Versions through v49 were titled *A 15-Qubit Decoherence-Reversal Protocol for Benchmarking Mid-Circuit Measurements and Environmental Imprinting*.  
**Zenodo:** [10.5281/zenodo.19105460](https://doi.org/10.5281/zenodo.19105460)  
**OSF:** [osf.io/9dnm8](https://osf.io/9dnm8)  
**arXiv:** [link to be added]

---

## Overview

This repository contains the implementation of a 15-qubit dynamic-circuit benchmark that stays small enough (2^15 amplitudes) for exact state-vector simulation, including every mid-circuit measurement branch. Because the ideal reference value is computed without approximation, any gap between the noise-model prediction and the observed result reflects the noise model or the execution stack, not truncation error in the reference calculation.

The circuit has three stages:
1. **Forward evolution:** a 15-layer scrambling circuit on a 6-qubit system, CNOT coupling to a 5-qubit engineered environment, and weak controlled-R_y coupling (λ = 0.15) to 4 probe qubits.
2. **Mid-circuit measurement:** the 4 probe qubits are read out mid-circuit and then discarded.
3. **Reversal:** the scrambling and system–environment coupling are inverted (the probe coupling is not). Recovery fidelity is measured on the 11-qubit system + environment register, with the probes traced out.

Three quantities are compared on the same circuit:
- **F_max**, the exact ideal recovery, computed by branch summation over all 16 probe outcomes.
- **F_expected**, a prediction from a local surrogate noise model built from published Quantinuum H2 headline error rates.
- **F_recovered**, the observed value from the Quantinuum H2-1 emulator.

F_max is below 1 even without noise, because the probe coupling is never undone and the information it carries leaves with the discarded probes. Measuring the probes before discarding them does not change the ideal value.

---

## Key Results (v50)

| Metric | Value |
|--------|-------|
| F_max (exact ideal baseline) | 0.9889 |
| F_expected (H2 Typical surrogate) | 0.8353 ± 0.0054 |
| Raw target-register ground-state frequency | 310 / 400 = 77.5% (inferred from F_recovered; the archived log lists only the top 10 bitstrings) |
| F_recovered (readout-mitigated, 400-shot emulator) | 0.7836 ± 0.0212 |
| Nominal gap | 0.0517, z_nom = 2.36 |

**Interpretation:** this is a *nominal surrogate-model mismatch*. The emulator returned lower recovery than a three-parameter surrogate predicts, by more than that surrogate's sampling error alone explains. This is expected: the H2-1E emulator models many noise sources (asymmetric readout, measurement and initialization crosstalk, spontaneous emission, dephasing) that the surrogate omits.

The z-score is conditional on the Typical surrogate being the correct model. It excludes noise-model, calibration, and run-to-run uncertainty, and the 2σ threshold was not fixed in advance. The observed value also lies between the Typical (0.8353) and Pessimistic (0.6805) predictions for the same circuit.

All results come from a single run on an **emulator**. No claim is made about decoherence on physical hardware. See the paper's Limitations section for the full list, including the noiseless-emulator control that has not yet been run.

The emulator job: `quantinuum.sim.h2-1e`, Job ID `cc0d204a-2346-11f1-8a26-0242ac1c000c`, submitted 19 March 2026 (UTC).

---

## Surrogate Sensitivity Sweep (Table 2)

| Scenario | Depth d | λ | (e1, e2, e_SPAM) | F_clean ± SE | F_exp ± SE | δ |
|----------|---|---|---|-------------|-----------|---|
| H2 Typical (production circuit) | 15 | 0.15 | (3e-5, 1e-3, 1e-3) | 0.9884 ± 0.0014 | 0.8353 ± 0.0054 | 0.1531 |
| H2 Pessimistic (production circuit) | 15 | 0.15 | (2e-4, 2e-3, 5e-3) | 0.9880 ± 0.0013 | 0.6805 ± 0.0071 | 0.3075 |
| Deep Scramble (different circuit) | 20 | 0.15 | (3e-5, 1e-3, 1e-3) | 0.9902 ± 0.0013 | 0.7831 ± 0.0064 | 0.2071 |
| Strong Probe Coupling (different circuit) | 15 | 0.30 | (3e-5, 1e-3, 1e-3) | 0.9556 ± 0.0029 | 0.8058 ± 0.0055 | 0.1498 |

Only the first two rows describe the production circuit; the other two show sensitivity to depth and probe coupling. Standard errors reflect sampling only (5,000 shots, 200 bootstrap resamples), not uncertainty in the noise model.

---

## Repository Contents

| File | Description |
|------|-------------|
| `15-Qubit Exact Simulation Protocol (v50).pdf` | Current manuscript (earlier versions, including v49, are archived in the [Zenodo version history](https://doi.org/10.5281/zenodo.19105460)) |
| `A 15-Qubit Exact-Simulation Protocol (v48) - Results.pdf` | Raw output log of the 400-shot emulator run |
| `local_simulation_(v48_master).py`, `Local_Simulation_(v48_Master).ipynb` | Local Aer simulation: computes F_max, runs the Table 2 sweep, and re-applies the decision rule to the observed emulator result |
| `azure_quantinuum_submission_(v48_public).py`, `Azure_Quantinuum_Submission_(v48_Public).ipynb` | Azure Quantum submission: runs the sweep, submits the 400-shot job to the H2-1E emulator via OpenQASM 2.0, and analyzes the result |
| `15-Qubit Loschmidt-Echo Benchmark Infographic (v50).png` | One-page visual summary (replaces the earlier "Decoherence-Reversal Protocol" infographic) |

### Note on the v48 code

The scripts are kept exactly as they were when they produced the published numbers. Some of their printed text predates the v50 interpretation:
- They label the outcome **"Anomalous Decoherence"**. The v50 paper describes it as a nominal surrogate-model mismatch on an emulator (see above).
- Their closing comments attribute the gap to compilation/routing overhead. The paper withdraws that explanation: H2 transports ions between gate zones and does not rely on logical SWAP insertion.
- The scenario printed as "Strong QND Kick" is called "Strong Probe Coupling" in the paper.
- One comment in the local-simulation notebook calls the emulator result a "physical run." It was an emulator run.

### Note on Register Structure

The two scripts use different classical register layouts by design:

- **Local simulation** uses three named registers (`cp`, `cs`, `ce`), which are readable, Qiskit-native and compatible with Aer.
- **Azure submission** uses a single flat 15-bit register, required for deterministic bitstring concatenation in the Quantinuum JSON payload.

Both scripts build the same circuit. The register difference is purely a classical-infrastructure adaptation.

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
python "local_simulation_(v48_master).py"
```

No Azure account is required. This computes F_max, runs the Table 2 sweep on the local Aer simulator, and applies the decision rule to the observed emulator result. Expect approximately 30 minutes on a standard Colab instance.

---

## Running on Azure Quantum (Quantinuum Emulator)

```bash
export AZURE_QUANTUM_RESOURCE_ID="<your-resource-id>"
export AZURE_TENANT_ID="<your-tenant-id>"
python "azure_quantinuum_submission_(v48_public).py"
```

You will be prompted to authenticate via browser (device code flow).

### Azure QIR Bypass

In our tests, the standard `backend.run()` path through the Azure QIR compiler did not return correct per-shot results for this multi-shot circuit with mid-circuit measurements. The script bypasses the QIR compiler by:

1. Compiling the circuit locally to OpenQASM 2.0 via `qasm2.dumps()`
2. Submitting via the `Job.from_input_data()` API with Honeywell format tags
3. Passing `input_params={"count": hardware_shots}` directly in the payload

**Estimated cost:** about 221 eHQC for a 400-shot run on `quantinuum.sim.h2-1e` (5 eHQC base fee plus about 0.54 eHQC per shot).

### Noiseless-emulator control (not yet run)

To run the same circuit on H2-1E with its noise model disabled, add `"error-model": False` to `input_params`:

```python
input_params={"count": cfg.hardware_shots, "error-model": False}
```

A result near F_max = 0.9889 would verify the submission path, register handling and mid-circuit measurement semantics end to end. Expect roughly the same eHQC cost as the noisy run.

---

## Citation

If you use this protocol or code, please cite:

```
Kahng, D. (2026). An Exactly Simulable 15-Qubit Loschmidt-Echo Benchmark
for Mid-Circuit Measurements and Noise-Model Validation (v50).
Zenodo. https://doi.org/10.5281/zenodo.19105460
```

---

## License

MIT License. See LICENSE file for details.

---

*AI Use Disclosure: The associated manuscript incorporates structural and drafting assistance from large language models. The author remains fully responsible for all content, theoretical claims, and scientific accuracy.*
