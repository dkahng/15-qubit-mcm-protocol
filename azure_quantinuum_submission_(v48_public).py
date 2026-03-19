# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Azure Quantinuum Submission (v48 Public Release - 400-Shot Validation)

Metadata Sync (v48):
- Production Target: hardware_shots set to 400 for final emulator validation.
- Surrogate Calibration: Utilizes published Quantinuum H2-1 Typical and Pessimistic error profiles.
- Infrastructure Bypass: Utilizes Job.from_input_data() with raw OpenQASM 2.0 and honeywell
  format tags to bypass Azure QIR compiler based on inferred limitations on mid-circuit measurements.
- Target: Defaults to Emulator (h2-1e). Change to 'quantinuum.qpu.h2-1' for physical QPU.
"""

import os
import sys
import subprocess
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Dependency check & Auto-Installer
# -----------------------------------------------------------------------------
try:
    from azure.identity import DeviceCodeCredential
    from azure.quantum import Workspace, Job
    from azure.quantum.qiskit import AzureQuantumProvider
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile, qasm2
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
except ImportError:
    print("[*] Installing missing Azure/Qiskit libraries...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "qiskit", "qiskit-aer", "azure-quantum[qiskit]", "azure-identity"
    ])
    from azure.identity import DeviceCodeCredential
    from azure.quantum import Workspace, Job
    from azure.quantum.qiskit import AzureQuantumProvider
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile, qasm2
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# -----------------------------------------------------------------------------
# Config & Models
# -----------------------------------------------------------------------------
# USER IDENTIFIERS STRIPPED FOR PUBLIC RELEASE
RESOURCE_ID = os.getenv("AZURE_QUANTUM_RESOURCE_ID", "<YOUR_AZURE_RESOURCE_ID>")
LOCATION = os.getenv("AZURE_QUANTUM_LOCATION", "eastus")
TARGET_NAME = os.getenv("AZURE_QUANTUM_TARGET", "quantinuum.sim.h2-1e")
TENANT_ID = os.getenv("AZURE_TENANT_ID", "<YOUR_AZURE_TENANT_ID>")


@dataclass(frozen=True)
class ProtocolConfig:
    seed: int = 42
    hardware_shots: int = 400          # 400-shot production validation
    sweep_shots: int = 5000
    bootstrap_reps_sweep: int = 200
    bootstrap_reps_hardware: int = 400
    probe_map: Tuple[Tuple[int, ...], ...] = field(
        default_factory=lambda: ((0, 1), (2, 3), (4, 5), (0, 5))
    )


@dataclass(frozen=True)
class DeviceCalibration:
    name: str; e1: float; e2: float; e_spam: float
    readout_p01: float | None = None
    readout_p10: float | None = None

    def matrices(self, n: int) -> List[np.ndarray]:
        p01 = self.e_spam if self.readout_p01 is None else self.readout_p01
        p10 = self.e_spam if self.readout_p10 is None else self.readout_p10
        M = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)
        return [M.copy() for _ in range(n)]


@dataclass(frozen=True)
class SweepScenario:
    name: str; depth: int; lambda_angle: float; calibration: DeviceCalibration


# -----------------------------------------------------------------------------
# Logic Core
# -----------------------------------------------------------------------------
def parse_bits(k: str) -> Tuple[str, str, str]:
    raw = k.replace(" ", "").zfill(15)
    # Qiskit big-endian: c14...c0
    probe = raw[0:4][::-1]    # c14..c11 -> p0..p3
    env   = raw[4:9][::-1]    # c10..c6  -> e0..e4
    sys   = raw[9:15][::-1]   # c5..c0   -> s0..s5
    return env, sys, probe


def get_angles(cfg: ProtocolConfig, d: int) -> List[List[float]]:
    rng = np.random.default_rng(cfg.seed)
    return [[rng.uniform(0.0, 2.0 * np.pi) for _ in range(6)] for _ in range(d)]


def build_full(cfg: ProtocolConfig, d: int, lam: float) -> QuantumCircuit:
    angles = get_angles(cfg, d)
    s, e, p = list(range(6)), list(range(6, 11)), list(range(11, 15))
    qr = QuantumRegister(15, "q")
    cr = ClassicalRegister(15, "c")
    full = QuantumCircuit(qr, cr, name="kahng_protocol_v48")
    # Forward Scrambling
    for layer in angles:
        for i, q in enumerate(s): full.ry(layer[i], qr[q])
        for q in range(5): full.cx(qr[s[q]], qr[s[q + 1]])
        full.cx(qr[s[5]], qr[s[0]])
    for i in range(5): full.cx(qr[s[i]], qr[e[i]])
    for i, controls in enumerate(cfg.probe_map):
        for ctrl in controls: full.cry(lam, qr[ctrl], qr[p[i]])
    full.measure(qr[p], cr[11:15])
    # Protocol Reversal
    for i in reversed(range(5)): full.cx(qr[s[i]], qr[e[i]])
    for layer in reversed(angles):
        full.cx(qr[s[5]], qr[s[0]])
        for q in reversed(range(5)): full.cx(qr[s[q]], qr[s[q + 1]])
        for i, q in enumerate(s): full.ry(-layer[i], qr[q])
    full.measure(qr[s], cr[0:6])
    full.measure(qr[e], cr[6:11])
    return full

def build_pure_fwd_rev(cfg, d, lam):
    angles = get_angles(cfg, d)
    fwd = QuantumCircuit(15)
    s, e, p = list(range(6)), list(range(6, 11)), list(range(11, 15))
    for layer in angles:
        for i, q in enumerate(s): fwd.ry(layer[i], q)
        for q in range(5): fwd.cx(s[q], s[q+1])
        fwd.cx(s[5], s[0])
    for i in range(5): fwd.cx(s[i], e[i])
    for i, controls in enumerate(cfg.probe_map):
        for ctrl in controls: fwd.cry(lam, ctrl, p[i])
    rev = QuantumCircuit(15)
    for i in reversed(range(5)): rev.cx(s[i], e[i])
    for layer in reversed(angles):
        rev.cx(s[5], s[0])
        for q in reversed(range(5)): rev.cx(s[q], s[q+1])
        for i, q in enumerate(s): rev.ry(-layer[i], q)
    return fwd, rev

def compute_fmax(cfg: ProtocolConfig, d: int, lam: float) -> float:
    fwd, rev = build_pure_fwd_rev(cfg, d, lam)
    psi_f = Statevector(fwd).data
    indices = np.arange(2**15, dtype=np.uint64)
    f_max = 0.0
    for val in range(16):
        mask = np.ones(2**15, dtype=bool)
        for i, q in enumerate(range(11, 15)): mask &= (((indices >> q) & 1) == ((val >> i) & 1))
        proj = np.where(mask, psi_f, 0.0j); pk = float(np.vdot(proj, proj).real)
        if pk > 1e-12:
            psi_rev = Statevector(proj / np.sqrt(pk)).evolve(rev).data
            f_max += pk * (np.abs(psi_rev[val << 11]) ** 2)
    return f_max

def mitigate(counts: Dict[str, int], mats: List[np.ndarray]) -> float:
    total = sum(counts.values())
    if total == 0: return 0.0
    coeffs = [(float(np.linalg.inv(M)[0, 0]), float(np.linalg.inv(M)[0, 1])) for M in mats]
    res = 0.0
    for key, count in counts.items():
        e, s, _ = parse_bits(key); target = s + e; term = 1.0
        for bit, (c0, c1) in zip(target, coeffs): term *= (c0 if bit == "0" else c1)
        res += (count / total) * term
    return max(0.0, min(1.0, res))

def bootstrap(counts: Dict[str, int], mats: List[np.ndarray], reps: int, seed: int) -> float:
    total = sum(counts.values())
    if total == 0 or reps < 2: return 0.0
    keys, probs = list(counts.keys()), np.array(list(counts.values()), dtype=float) / total
    rng = np.random.default_rng(seed)
    ests = [mitigate({k: int(c) for k, c in zip(keys, rng.multinomial(total, probs)) if c > 0}, mats) for _ in range(reps)]
    return float(np.std(ests, ddof=1))


# -----------------------------------------------------------------------------
# Main Runtime
# -----------------------------------------------------------------------------
def main():
    print("=== 15-Qubit Protocol: Azure Submission Workflow (v48 Public) ===")
    cfg = ProtocolConfig()

    print("\n[Phase I] Connecting to Azure Quantum & Defining Calibrations...")
    try:
        ws = Workspace(resource_id=RESOURCE_ID, location=LOCATION, credential=DeviceCodeCredential(tenant_id=TENANT_ID))
        backend = AzureQuantumProvider(workspace=ws).get_backend(TARGET_NAME)
        print(f"    Target Backend: {backend.name}")
    except Exception as e: print(f"[-] Connection failed: {e}"); return

    print("\n[Phase II] Computing Analytical F_max baseline...")
    f_max = compute_fmax(cfg, 15, 0.15)
    print(f"    F_max Limit: {f_max:.4f}")

    print("\n[Support Analysis] Sensitivity Sweep (Table 2)")
    print(f"    {'Scenario':<20} | {'F_clean ± SE':<18} | {'F_expected ± SE':<18}")
    print("    " + "-" * 65)

    scenarios = [
        SweepScenario("H2 Typical Profile",   15, 0.15, DeviceCalibration("H2_Typ",  3e-5, 1e-3, 1e-3)),
        SweepScenario("H2 Pessimistic Limit", 15, 0.15, DeviceCalibration("H2_Pess", 2e-4, 2e-3, 5e-3)),
        SweepScenario("Deep Scramble (H2)",   20, 0.15, DeviceCalibration("H2_Typ",  3e-5, 1e-3, 1e-3)),
        SweepScenario("Strong QND Kick (H2)", 15, 0.30, DeviceCalibration("H2_Typ",  3e-5, 1e-3, 1e-3)),
    ]

    primary_data = None
    for s in scenarios:
        qc = build_full(cfg, s.depth, s.lambda_angle)
        comp = transpile(qc, backend, optimization_level=3)
        comp.data = [inst for inst in comp.data if inst.operation.name != "barrier"]
        sim = AerSimulator(); t_sim = transpile(comp, sim, optimization_level=0)
        c_cts = sim.run(t_sim, shots=cfg.sweep_shots).result().get_counts()
        f_c = mitigate(c_cts, [np.eye(2) for _ in range(11)])
        se_c = bootstrap(c_cts, [np.eye(2) for _ in range(11)], cfg.bootstrap_reps_sweep, cfg.seed + 1)
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(s.calibration.e1, 1), ['ry', 'rx', 'rz', 'sx', 'x'])
        noise.add_all_qubit_quantum_error(depolarizing_error(s.calibration.e2, 2), ['cx', 'cry', 'cz', 'rzz'])
        mats = s.calibration.matrices(11)
        for q, M in enumerate(mats): noise.add_readout_error(ReadoutError(M.tolist()), [q])
        n_cts = AerSimulator(noise_model=noise).run(t_sim, shots=cfg.sweep_shots).result().get_counts()
        f_e = mitigate(n_cts, mats); se_e = bootstrap(n_cts, mats, cfg.bootstrap_reps_sweep, cfg.seed + 2)
        print(f"    {s.name:<20} | {f_c:.4f} ± {se_c:.4f}   | {f_e:.4f} ± {se_e:.4f}")

        if s.name == "H2 Typical Profile": primary_data = (comp, f_e, se_e)

    print(f"\n[Phase V] Submitting OpenQASM 2.0 Payload via Bare-Metal API ({cfg.hardware_shots} shots)...")
    try:
        qasm_str = qasm2.dumps(primary_data[0])
        job = Job.from_input_data(
            workspace=ws,
            name="kahng_v48_public_run",
            target=TARGET_NAME,
            input_data=qasm_str.encode('utf-8'),
            content_type="application/qasm",
            provider_id="quantinuum",
            input_data_format="honeywell.openqasm.v1",
            output_data_format="honeywell.quantum-results.v1",
            input_params={"count": cfg.hardware_shots}
        )
        print(f"    Job ID: {job.id}")
        job.wait_until_completed(max_poll_wait_secs=20, timeout_secs=1800, print_progress=True)
        job.refresh()
        print(f"    Final status: {job.details.status!r}")
    except Exception as e:
        print(f"[-] Submission failed: {e}"); return

    print("\n[Phase VI] Result extraction and analysis")
    try:
        results = job.get_results(timeout_secs=300)
        if isinstance(results, dict) and results:
            reg_name, shots_list = None, None
            for k, v in results.items():
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], str)):
                    reg_name, shots_list = k, v; break
            if shots_list is None:
                print(f"[-] No bitstring list found: {results!r}"); return
            cts = dict(Counter(shots_list)); total_returned = sum(cts.values())
            print(f"    [*] Register: {reg_name!r} | Shots: {total_returned}")
            print(f"    [*] Counts (top 10): {dict(list(cts.items())[:10])}")

            if total_returned != cfg.hardware_shots:
                print(f"    [!] Warning: Requested {cfg.hardware_shots}, got {total_returned}")

            f_exp, se_exp = primary_data[1], primary_data[2]
            mats = scenarios[0].calibration.matrices(11)
            f_rec = mitigate(cts, mats); se_rec = bootstrap(cts, mats, cfg.bootstrap_reps_hardware, cfg.seed + 3)
            band = 2.0 * np.sqrt(se_rec**2 + se_exp**2)

            # THREE-WAY INTERPRETATION RULE
            if abs(f_rec - f_exp) <= band:
                interp = "Consistent with modeled reversal"
            elif f_rec < (f_exp - band):
                interp = "Anomalous Decoherence"
            else:
                interp = "Inconclusive"

            print(f"    F_max (Ideal limit)           : {f_max:.4f}")
            print(f"    F_expected (Surrogate)        : {f_exp:.4f} ± {se_exp:.4f}")
            print(f"    F_recovered (Physical/Emul.)  : {f_rec:.4f} ± {se_rec:.4f}")
            print(f"    Interpretation (Table 3 rule) : {interp}")
        else:
            print(f"[-] Unexpected result payload: {results!r}")
    except Exception as e:
        print(f"[-] Data extraction/analysis failed: {e}")

if __name__ == "__main__":
    main()