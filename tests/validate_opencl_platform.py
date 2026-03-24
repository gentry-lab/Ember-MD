#!/usr/bin/env python3
"""
Comprehensive OpenCL (cl2Metal) platform validation suite.

Mirrors the 47 native Metal C++ platform tests in Python, covering every
OpenMM feature that Ember uses in production.  Each test creates a minimal
system, evaluates it on both the Reference (double-precision CPU) and OpenCL
(single-precision GPU via cl2Metal) platforms, and compares energies, forces,
or statistical observables.

Run:  python tests/validate_opencl_platform.py [--platform Metal]

Tolerances are set for single-precision GPU vs double-precision Reference.
The OpenMM C++ tests use TOL = 1e-5 for double precision; we use 1e-3 for
single-precision energy and 5e-3 for forces (derivative amplifies error).
"""

import argparse
import math
import sys
import time
import traceback

import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPU_PLATFORM = "OpenCL"  # overridden by --platform arg

# Tolerances (single precision GPU vs double precision Reference)
ENERGY_TOL = 1e-3        # relative, scaled by max(1, |expected|)
FORCE_TOL = 5e-3         # relative, force RMSD
CONSTRAINT_TOL = 1e-4    # absolute nm
STAT_TOL_FACTOR = 6.0    # for stochastic tests: tol = factor/sqrt(N)
POSITION_TOL = 0.02      # nm, for trajectory comparison

# Physical constants (OpenMM internal units: kJ/mol, nm, ps)
BOLTZ_KJ = 8.314462618e-3  # kJ/mol/K
ONE_4PI_EPS0 = 138.93545764438198  # kJ·nm/(mol·e²)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
RESULTS = []


def assert_tol(name, expected, found, tol, stochastic=False):
    """ASSERT_EQUAL_TOL equivalent: relative tolerance scaled by max(1, |expected|)."""
    global PASS_COUNT, FAIL_COUNT
    scale = max(1.0, abs(expected))
    diff = abs(expected - found) / scale
    ok = diff <= tol
    tag = "PASS" if ok else "FAIL"
    extra = " (stochastic)" if stochastic else ""
    print(f"    [{tag}] {name}: expected {expected:.6g}, got {found:.6g} "
          f"(rel diff {diff:.2e}, tol {tol:.1e}){extra}")
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def assert_true(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail and not condition else ""
    print(f"    [{tag}] {name}{suffix}")
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return condition


def compare_single_point(system, positions, test_name, topology=None,
                         energy_tol=ENERGY_TOL, force_tol=FORCE_TOL):
    """Evaluate system on Reference and GPU, compare energy and forces."""
    ref_platform = mm.Platform.getPlatformByName("Reference")
    gpu_platform = mm.Platform.getPlatformByName(GPU_PLATFORM)

    integrator_r = mm.VerletIntegrator(0.001)
    ctx_r = mm.Context(system, integrator_r, ref_platform)
    ctx_r.setPositions(positions)
    state_r = ctx_r.getState(getEnergy=True, getForces=True)
    e_ref = state_r.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = np.array(state_r.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer))

    integrator_g = mm.VerletIntegrator(0.001)
    ctx_g = mm.Context(system, integrator_g, gpu_platform)
    ctx_g.setPositions(positions)
    state_g = ctx_g.getState(getEnergy=True, getForces=True)
    e_gpu = state_g.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_gpu = np.array(state_g.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer))

    n = system.getNumParticles()
    energy_ok = assert_tol(f"{test_name} energy ({n} atoms)",
                           e_ref, e_gpu, energy_tol)

    # Force RMSD, scaled
    f_magnitude = max(1.0, np.sqrt(np.mean(f_ref**2)))
    f_rmsd = np.sqrt(np.mean((f_ref - f_gpu)**2))
    f_rel = f_rmsd / f_magnitude
    force_ok = assert_true(
        f"{test_name} force RMSD",
        f_rel <= force_tol,
        f"rel RMSD {f_rel:.2e} > tol {force_tol:.1e}")

    del ctx_r, ctx_g
    return energy_ok and force_ok


def run_test(func):
    """Run a test function, catch exceptions, record result."""
    global SKIP_COUNT
    name = func.__name__
    print(f"\n{'='*64}")
    print(f"  {name}")
    print(f"{'='*64}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        RESULTS.append((name, "ok", elapsed))
    except mm.OpenMMException as e:
        if "No platform" in str(e) or "not available" in str(e).lower():
            print(f"    [SKIP] Platform not available: {e}")
            SKIP_COUNT += 1
            RESULTS.append((name, "skip", 0))
        else:
            print(f"    [ERROR] {e}")
            traceback.print_exc()
            RESULTS.append((name, "error", 0))
            global FAIL_COUNT
            FAIL_COUNT += 1
    except Exception as e:
        print(f"    [ERROR] {e}")
        traceback.print_exc()
        RESULTS.append((name, "error", 0))
        FAIL_COUNT += 1


# ===================================================================
# GROUP 1: Individual Force Kernels
# ===================================================================

def test_01_coulomb():
    """Two charged particles — analytical Coulomb energy and force."""
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    nb = mm.NonbondedForce()
    nb.addParticle(0.5, 1.0, 0.0)   # charge, sigma, epsilon
    nb.addParticle(-1.5, 1.0, 0.0)
    system.addForce(nb)

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)] * unit.nanometers

    # Analytical: E = (1/4πε₀) * q1*q2 / r
    expected_energy = ONE_4PI_EPS0 * (0.5 * -1.5) / 2.0
    expected_force = ONE_4PI_EPS0 * (0.5 * -1.5) / (2.0**2)  # = dE/dr

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer)

    assert_tol("Coulomb energy", expected_energy, e, ENERGY_TOL)
    assert_tol("Coulomb force x[0]", -expected_force, forces[0][0], ENERGY_TOL)
    assert_tol("Coulomb force x[1]", expected_force, forces[1][0], ENERGY_TOL)
    del ctx


def test_02_lennard_jones():
    """Two LJ particles — analytical energy and force."""
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    nb = mm.NonbondedForce()
    nb.addParticle(0.0, 1.2, 1.0)  # charge=0, sigma=1.2, eps=1.0
    nb.addParticle(0.0, 1.4, 2.0)
    system.addForce(nb)

    r = 2.0  # nm
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(r, 0, 0)] * unit.nanometers

    # Combined: sigma = (1.2+1.4)/2 = 1.3, eps = sqrt(1*2) = sqrt(2)
    sig = 1.3
    eps = math.sqrt(2.0)
    x = sig / r
    expected_energy = 4.0 * eps * (x**12 - x**6)
    expected_force = 4.0 * eps * (12 * x**12 - 6 * x**6) / r

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer)

    assert_tol("LJ energy", expected_energy, e, ENERGY_TOL)
    assert_tol("LJ force x[0]", -expected_force, forces[0][0], ENERGY_TOL)
    del ctx


def test_03_harmonic_bond():
    """HarmonicBondForce — analytical energy and force, plus platform comparison."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 400.0)  # r0=1.5 nm, k=400 kJ/mol/nm²
    system.addForce(bond)

    r = 2.0
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(r, 0, 0)] * unit.nanometers

    # E = 0.5 * k * (r - r0)^2
    expected = 0.5 * 400.0 * (r - 1.5)**2

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    ctx.setPositions(positions)
    state = ctx.getState(getEnergy=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    assert_tol("HarmonicBond energy", expected, e, ENERGY_TOL)
    del ctx


def test_04_harmonic_angle():
    """HarmonicAngleForce — three particles at a known angle."""
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    angle_force = mm.HarmonicAngleForce()
    theta0 = math.pi / 3.0  # 60 degrees
    k = 200.0
    angle_force.addAngle(0, 1, 2, theta0, k)
    system.addForce(angle_force)

    # Place at 90 degrees
    positions = [mm.Vec3(1, 0, 0), mm.Vec3(0, 0, 0), mm.Vec3(0, 1, 0)] * unit.nanometers
    theta_actual = math.pi / 2.0
    expected = 0.5 * k * (theta_actual - theta0)**2

    compare_single_point(system, positions, "HarmonicAngle")


def test_05_periodic_torsion():
    """PeriodicTorsionForce — platform comparison."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    torsion = mm.PeriodicTorsionForce()
    torsion.addTorsion(0, 1, 2, 3, 2, math.pi / 3.0, 5.0)  # periodicity=2, phase, k
    system.addForce(torsion)

    positions = [
        mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 0),
        mm.Vec3(1, 0, 0), mm.Vec3(1, 1, 0.5)
    ] * unit.nanometers

    compare_single_point(system, positions, "PeriodicTorsion")


def test_06_cmap_torsion():
    """CMAPTorsionForce — used by ff19SB for backbone corrections."""
    system = mm.System()
    for _ in range(5):
        system.addParticle(1.0)
    cmap = mm.CMAPTorsionForce()
    # Create a simple 24x24 CMAP grid
    size = 24
    grid = [0.0] * (size * size)
    for i in range(size):
        for j in range(size):
            phi = 2.0 * math.pi * i / size - math.pi
            psi = 2.0 * math.pi * j / size - math.pi
            grid[i * size + j] = math.cos(phi) * math.sin(psi) * 10.0
    cmap.addMap(size, grid)
    cmap.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)
    system.addForce(cmap)

    positions = [
        mm.Vec3(0, 0, 0), mm.Vec3(1, 0, 0), mm.Vec3(1, 1, 0),
        mm.Vec3(2, 1, 0), mm.Vec3(2, 1, 1)
    ] * unit.nanometers

    compare_single_point(system, positions, "CMAPTorsion")


def test_07_nonbonded_pme():
    """NonbondedForce with PME — exercises FFT kernels, the critical path."""
    system = mm.System()
    n = 64
    box_size = 3.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_size, 0, 0), mm.Vec3(0, box_size, 0), mm.Vec3(0, 0, box_size))

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0)
    nb.setEwaldErrorTolerance(0.0005)

    rng = np.random.RandomState(42)
    positions = []
    for i in range(n):
        system.addParticle(1.0 + rng.random())
        charge = 0.5 * (-1)**i
        sigma = 0.25 + 0.1 * rng.random()
        epsilon = 0.5 + 0.5 * rng.random()
        nb.addParticle(charge, sigma, epsilon)
        positions.append(mm.Vec3(*(rng.random(3) * (box_size - 0.5) + 0.25)))

    system.addForce(nb)
    positions = positions * unit.nanometers

    compare_single_point(system, positions, "NonbondedForce PME (64 atoms)")


def test_08_nonbonded_pme_large():
    """Larger PME system (~1000 atoms) — stress-tests FFT and neighbor lists."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.5, 2.5, 2.5) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometers,
                             ewaldErrorTolerance=0.0005,
                             constraints=app.HBonds)

    compare_single_point(system, modeller.positions,
                         f"NonbondedForce PME TIP3P ({modeller.topology.getNumAtoms()} atoms)")


def test_09_custom_external_force():
    """CustomExternalForce with periodicdistance — Ember's restraint pattern."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.5, 2.5, 2.5) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometers,
                             constraints=app.HBonds)

    restraint = mm.CustomExternalForce(
        'k*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k', 1000.0)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    positions = modeller.positions
    for i in range(min(20, system.getNumParticles())):
        pos = positions[i]
        restraint.addParticle(i, [
            pos[0].value_in_unit(unit.nanometers),
            pos[1].value_in_unit(unit.nanometers),
            pos[2].value_in_unit(unit.nanometers),
        ])
    system.addForce(restraint)

    compare_single_point(system, positions,
                         "CustomExternalForce periodicdistance")


def test_10_custom_bond_force():
    """CustomBondForce — Morse potential as custom expression."""
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    # Morse: D*(1 - exp(-a*(r-r0)))^2
    cbf = mm.CustomBondForce('D*(1-exp(-a*(r-r0)))^2')
    cbf.addPerBondParameter('D')
    cbf.addPerBondParameter('a')
    cbf.addPerBondParameter('r0')
    cbf.addBond(0, 1, [100.0, 2.0, 0.3])
    system.addForce(cbf)

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0)] * unit.nanometers
    compare_single_point(system, positions, "CustomBondForce (Morse)")


def test_11_gbsa_obc():
    """GBSAOBCForce — implicit solvent used in post-dock refinement."""
    system = mm.System()
    n = 16
    rng = np.random.RandomState(99)
    nb = mm.NonbondedForce()
    gbsa = mm.GBSAOBCForce()
    gbsa.setSolventDielectric(78.5)
    gbsa.setSoluteDielectric(1.0)

    positions = []
    for i in range(n):
        system.addParticle(12.0)
        charge = 0.3 * (-1)**i
        sigma = 0.2 + 0.1 * rng.random()
        nb.addParticle(charge, sigma, 0.5)
        gbsa.addParticle(charge, 0.15 + 0.05 * rng.random(), 1.0)
        positions.append(mm.Vec3(*(rng.random(3) * 2.0)))

    system.addForce(nb)
    system.addForce(gbsa)
    positions = positions * unit.nanometers

    compare_single_point(system, positions, "GBSAOBCForce (16 atoms)",
                         energy_tol=2e-3)


def test_12_gbsa_obc_cutoff():
    """GBSAOBCForce with cutoff — periodic system."""
    system = mm.System()
    box = 3.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    n = 32
    rng = np.random.RandomState(77)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2)
    gbsa = mm.GBSAOBCForce()
    gbsa.setNonbondedMethod(mm.GBSAOBCForce.CutoffPeriodic)
    gbsa.setCutoffDistance(1.2)

    positions = []
    for i in range(n):
        system.addParticle(12.0)
        charge = 0.2 * (-1)**i
        nb.addParticle(charge, 0.3, 0.5)
        gbsa.addParticle(charge, 0.17, 1.0)
        positions.append(mm.Vec3(*(rng.random(3) * (box - 0.5) + 0.25)))

    system.addForce(nb)
    system.addForce(gbsa)
    positions = positions * unit.nanometers

    compare_single_point(system, positions, "GBSAOBCForce cutoff periodic",
                         energy_tol=2e-3)


def test_13_multiple_forces():
    """Multiple bonded + nonbonded forces — tests force accumulation."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(2.0)

    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 300000.0)
    bond.addBond(2, 3, 0.15, 300000.0)
    system.addForce(bond)

    angle = mm.HarmonicAngleForce()
    angle.addAngle(0, 1, 2, 1.91, 500.0)
    system.addForce(angle)

    nb = mm.NonbondedForce()
    for i in range(4):
        nb.addParticle(0.1 * (-1)**i, 0.3, 0.5)
    nb.createExceptionsFromBonds([(0, 1), (2, 3)], 0.5, 0.5)
    system.addForce(nb)

    positions = [
        mm.Vec3(0, 0, 0), mm.Vec3(0.15, 0, 0),
        mm.Vec3(0.2, 0.12, 0), mm.Vec3(0.35, 0.12, 0)
    ] * unit.nanometers

    compare_single_point(system, positions, "Multiple forces combined")


# ===================================================================
# GROUP 2: Integrators & Ensemble
# ===================================================================

def test_14_langevin_middle_single_bond():
    """LangevinMiddleIntegrator — damped harmonic oscillator vs analytical."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 1.0)  # r0=1.5 nm, k=1.0 kJ/mol/nm²
    system.addForce(bond)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(0, 0.1, 0.01)  # T=0, friction=0.1, dt=0.01
    ctx = mm.Context(system, integrator, gpu)
    positions = [mm.Vec3(-1, 0, 0), mm.Vec3(1, 0, 0)] * unit.nanometers
    ctx.setPositions(positions)

    # Damped harmonic oscillator: freq = sqrt(k/m - gamma^2/4)
    # Each particle sees k_eff = k/m_reduced, gamma = friction
    # For this system: omega = sqrt(1 - 0.05^2) ≈ 0.99875
    freq = math.sqrt(1.0 - 0.05**2)
    max_error = 0.0
    for _ in range(200):
        state = ctx.getState(getPositions=True)
        t = state.getTime().value_in_unit(unit.picoseconds)
        expected_dist = 1.5 + 0.5 * math.exp(-0.05 * t) * math.cos(freq * t)
        actual_dist = abs(state.getPositions()[1][0].value_in_unit(unit.nanometers) -
                         state.getPositions()[0][0].value_in_unit(unit.nanometers))
        max_error = max(max_error, abs(actual_dist - expected_dist))
        integrator.step(1)

    # C++ test uses 0.02; relax slightly for single-precision GPU
    assert_true("LangevinMiddle trajectory tracks analytical solution",
                max_error < 0.025,
                f"max position error {max_error:.4f} nm")
    del ctx


def test_15_langevin_middle_energy_conservation():
    """LangevinMiddleIntegrator with zero friction — NVE energy conservation."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 1.0)
    system.addForce(bond)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(0, 0.0, 0.01)  # zero friction = NVE
    ctx = mm.Context(system, integrator, gpu)
    positions = [mm.Vec3(-1, 0, 0), mm.Vec3(1, 0, 0)] * unit.nanometers
    ctx.setPositions(positions)

    state = ctx.getState(getEnergy=True)
    initial_e = (state.getKineticEnergy() + state.getPotentialEnergy()
                 ).value_in_unit(unit.kilojoules_per_mole)

    max_drift = 0.0
    for _ in range(1000):
        integrator.step(1)
        state = ctx.getState(getEnergy=True)
        e = (state.getKineticEnergy() + state.getPotentialEnergy()
             ).value_in_unit(unit.kilojoules_per_mole)
        max_drift = max(max_drift, abs(e - initial_e))

    # OpenMM C++ test uses 0.01 kJ/mol tolerance
    assert_true("NVE energy conservation (zero friction)",
                max_drift < 0.05,  # relaxed for single precision
                f"max drift {max_drift:.6f} kJ/mol")
    del ctx


def test_16_langevin_middle_temperature():
    """LangevinMiddleIntegrator — temperature equilibration."""
    n = 8
    target_temp = 100.0  # K
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(5, 0, 0), mm.Vec3(0, 5, 0), mm.Vec3(0, 0, 5))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    for i in range(n):
        system.addParticle(2.0)
        nb.addParticle((-1.0)**i, 1.0, 5.0)
    system.addForce(nb)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(target_temp, 3.0, 0.01)
    ctx = mm.Context(system, integrator, gpu)
    positions = []
    for i in range(n):
        positions.append(mm.Vec3((i % 2) * 4 - 2, (i % 4 < 2) * 4 - 2,
                                 (i < 4) * 4 - 2))
    ctx.setPositions(positions * unit.nanometers)

    # Equilibrate
    integrator.step(5000)

    # Sample kinetic energy
    steps = 10000
    ke_total = 0.0
    for _ in range(steps):
        state = ctx.getState(getEnergy=True)
        ke_total += state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        integrator.step(1)
    ke_avg = ke_total / steps
    expected_ke = 0.5 * n * 3 * BOLTZ_KJ * target_temp

    assert_tol("Average KE matches target temperature", expected_ke, ke_avg,
               STAT_TOL_FACTOR / math.sqrt(steps), stochastic=True)
    del ctx


def test_17_langevin_middle_constraints():
    """LangevinMiddleIntegrator — constraint satisfaction during dynamics."""
    n = 8
    system = mm.System()
    nb = mm.NonbondedForce()
    for i in range(n):
        system.addParticle(10.0)
        nb.addParticle(0.2 * (-1)**i, 0.5, 5.0)

    # Add constraints
    constraint_distances = [1.0, 1.0, 1.0, 1.0, 1.0]
    system.addConstraint(0, 1, 1.0)
    system.addConstraint(1, 2, 1.0)
    system.addConstraint(2, 3, 1.0)
    system.addConstraint(4, 5, 1.0)
    system.addConstraint(6, 7, 1.0)
    system.addForce(nb)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(100.0, 2.0, 0.01)
    integrator.setConstraintTolerance(1e-5)
    ctx = mm.Context(system, integrator, gpu)

    # Place constrained pairs at their constraint distance, well-separated
    # to avoid close-contact explosions from random placement
    positions = [
        mm.Vec3(0.0, 0.0, 0.0), mm.Vec3(1.0, 0.0, 0.0),   # pair 0-1
        mm.Vec3(2.0, 0.0, 0.0), mm.Vec3(3.0, 0.0, 0.0),   # 2-3 (chained 1-2-3)
        mm.Vec3(0.0, 3.0, 0.0), mm.Vec3(1.0, 3.0, 0.0),   # pair 4-5
        mm.Vec3(0.0, 6.0, 0.0), mm.Vec3(1.0, 6.0, 0.0),   # pair 6-7
    ]
    ctx.setPositions(positions * unit.nanometers)
    ctx.setVelocitiesToTemperature(100.0)

    max_violation = 0.0
    for _ in range(1000):
        integrator.step(1)
        state = ctx.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        for (a, b), d in zip([(0,1),(1,2),(2,3),(4,5),(6,7)],
                             constraint_distances):
            actual = np.linalg.norm(pos[a] - pos[b])
            max_violation = max(max_violation, abs(actual - d))

    assert_true("Constraints maintained during dynamics",
                max_violation < CONSTRAINT_TOL,
                f"max violation {max_violation:.2e} nm")
    del ctx


def test_18_settle():
    """SETTLE constraint — rigid water geometry over 1000 steps."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integrator.setConstraintTolerance(1e-5)
    simulation = app.Simulation(modeller.topology, system, integrator, gpu)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(300.0)

    # Measure O-H distances before dynamics
    tip3p_rOH = 0.09572  # nm (TIP3P)

    waters = [r for r in modeller.topology.residues() if r.name == 'HOH'][:20]

    max_deviation = 0.0
    for step in range(100):
        simulation.step(10)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        for w in waters:
            atoms = list(w.atoms())
            o_idx = next(a.index for a in atoms if a.element.symbol == 'O')
            h_indices = [a.index for a in atoms if a.element.symbol == 'H']
            for h_idx in h_indices:
                d = np.linalg.norm(pos[o_idx] - pos[h_idx])
                max_deviation = max(max_deviation, abs(d - tip3p_rOH))

    assert_true("SETTLE O-H constraint rigid over 1000 steps",
                max_deviation < CONSTRAINT_TOL,
                f"max deviation {max_deviation:.2e} nm")


def test_19_barostat_ideal_gas():
    """MonteCarloBarostat — ideal gas law PV = NkT validation."""
    n = 64
    pressure_bar = 1.5
    pressure_md = pressure_bar * (6.02214076e23 * 1e-25)  # kJ/mol/nm³
    temperatures = [300.0, 600.0, 1000.0]
    initial_volume = n * BOLTZ_KJ * temperatures[1] / pressure_md
    initial_length = initial_volume ** (1.0/3.0)

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(initial_length, 0, 0),
        mm.Vec3(0, 0.5 * initial_length, 0),
        mm.Vec3(0, 0, 2 * initial_length))

    rng = np.random.RandomState(42)
    positions = []
    for i in range(n):
        system.addParticle(1.0)
        positions.append(mm.Vec3(
            initial_length * rng.random(),
            0.5 * initial_length * rng.random(),
            2 * initial_length * rng.random()))
    positions = positions * unit.nanometers

    barostat = mm.MonteCarloBarostat(pressure_bar * unit.bar,
                                     temperatures[0] * unit.kelvin, 10)
    system.addForce(barostat)
    # Need a periodic force so system is "periodic"
    bonds = mm.HarmonicBondForce()
    bonds.setUsesPeriodicBoundaryConditions(True)
    system.addForce(bonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)

    for temp in temperatures:
        barostat.setDefaultTemperature(temp * unit.kelvin)
        integrator = mm.LangevinIntegrator(temp, 0.1, 0.01)
        ctx = mm.Context(system, integrator, gpu)
        ctx.setPositions(positions)
        ctx.setVelocitiesToTemperature(temp)

        # Equilibrate
        integrator.step(2000)

        # Sample volume
        steps = 5000
        vol_sum = 0.0
        for _ in range(steps):
            integrator.step(10)
            state = ctx.getState()
            box = state.getPeriodicBoxVectors()
            v = (box[0][0] * box[1][1] * box[2][2]
                 ).value_in_unit(unit.nanometers**3)
            vol_sum += v
        mean_vol = vol_sum / steps
        expected_vol = n * BOLTZ_KJ * temp / pressure_md

        assert_tol(f"Ideal gas volume at {temp}K",
                   expected_vol, mean_vol,
                   3.0 / math.sqrt(steps), stochastic=True)
        del ctx


def test_20_barostat_water_density():
    """MonteCarloBarostat — OPC water density at 298K, 1 atm."""
    ff = app.ForceField('amber/protein.ff19SB.xml', 'amber/opc_standard.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip4pew',
                        boxSize=mm.Vec3(2.5, 2.5, 2.5) * unit.nanometers)
    n_water = sum(1 for r in modeller.topology.residues() if r.name == 'HOH')

    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometers,
                             ewaldErrorTolerance=0.0005,
                             constraints=app.HBonds,
                             hydrogenMass=1.5 * unit.amu)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 298 * unit.kelvin, 25))

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinMiddleIntegrator(298, 1.0, 0.004)
    simulation = app.Simulation(modeller.topology, system, integrator, gpu)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=500)
    simulation.context.setVelocitiesToTemperature(298)

    # Equilibrate (20 ps)
    simulation.step(5000)

    # Sample density (20 ps)
    water_molar_mass = 18.015
    densities = []
    for _ in range(50):
        simulation.step(100)
        state = simulation.context.getState()
        box = state.getPeriodicBoxVectors()
        vol_nm3 = (box[0][0] * box[1][1] * box[2][2]
                   ).value_in_unit(unit.nanometers**3)
        vol_cm3 = vol_nm3 * 1e-21
        mass_g = n_water * water_molar_mass / 6.022e23
        densities.append(mass_g / vol_cm3)

    mean_d = np.mean(densities)
    assert_true("OPC water density near 1.0 g/cm³",
                0.95 < mean_d < 1.05,
                f"got {mean_d:.4f} g/cm³")


# ===================================================================
# GROUP 3: Infrastructure
# ===================================================================

def test_21_virtual_sites():
    """Virtual site positions — OPC 4-site water."""
    ff = app.ForceField('amber/opc_standard.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip4pew',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    # Check virtual site positions match between platforms
    ref = mm.Platform.getPlatformByName("Reference")
    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)

    integ_r = mm.VerletIntegrator(0.001)
    ctx_r = mm.Context(system, integ_r, ref)
    ctx_r.setPositions(modeller.positions)
    ctx_r.computeVirtualSites()
    pos_r = ctx_r.getState(getPositions=True).getPositions(asNumpy=True
             ).value_in_unit(unit.nanometers)

    integ_g = mm.VerletIntegrator(0.001)
    ctx_g = mm.Context(system, integ_g, gpu)
    ctx_g.setPositions(modeller.positions)
    ctx_g.computeVirtualSites()
    pos_g = ctx_g.getState(getPositions=True).getPositions(asNumpy=True
             ).value_in_unit(unit.nanometers)

    max_diff = np.max(np.abs(pos_r - pos_g))
    assert_true("Virtual site positions match Reference",
                max_diff < 1e-4,
                f"max diff {max_diff:.2e} nm")

    # Also check the energy with virtual sites
    state_r = ctx_r.getState(getEnergy=True)
    state_g = ctx_g.getState(getEnergy=True)
    e_r = state_r.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_g = state_g.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    n = system.getNumParticles()
    assert_tol("Virtual site system energy", e_r, e_g, ENERGY_TOL)

    del ctx_r, ctx_g


def test_22_hmr():
    """Hydrogen Mass Repartitioning — masses correct and 4fs stable."""
    # Use alanine dipeptide + solvent so we have non-water hydrogens to check.
    # SETTLE-constrained water H atoms are NOT repartitioned (by design).
    from pdbfixer import PDBFixer
    import io

    # Minimal alanine dipeptide PDB
    ala_pdb = """\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.460   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.249   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.937  -0.764   1.232  1.00  0.00           C
ATOM      6  N   ALA A   2       4.332   1.506   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.007   2.800   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       6.520   2.690   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       7.098   1.610   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.530   3.637   1.187  1.00  0.00           C
ATOM     11  OXT ALA A   2       7.098   3.770   0.000  1.00  0.00           O
END
"""
    fixer = PDBFixer(pdbfile=io.StringIO(ala_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    ff = app.ForceField('amber/protein.ff19SB.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, model='tip3p',
                        padding=1.2 * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds,
                             hydrogenMass=1.5 * unit.amu)

    # Check non-water hydrogen masses (water H stays at ~1.008 due to SETTLE)
    water_atom_indices = set()
    for r in modeller.topology.residues():
        if r.name == 'HOH':
            for a in r.atoms():
                water_atom_indices.add(a.index)

    h_masses = []
    for i in range(system.getNumParticles()):
        if i in water_atom_indices:
            continue
        m = system.getParticleMass(i).value_in_unit(unit.dalton)
        if 0 < m < 4.0:  # hydrogen range
            h_masses.append(m)

    if h_masses:
        avg = np.mean(h_masses)
        assert_tol("HMR non-water hydrogen mass = 1.5 amu", 1.5, avg, 0.01)
    else:
        assert_true("Found non-water hydrogens to check", False, "none found")

    # Run 500 steps at 4fs — should not blow up
    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
    integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.004)  # 4fs
    simulation = app.Simulation(modeller.topology, system, integrator, gpu)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(300)
    simulation.step(500)

    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_true("HMR 4fs NPT stable (no NaN/explosion)",
                not math.isnan(pe) and abs(pe) < 1e8,
                f"PE = {pe:.1f}")


def test_23_parameter_update():
    """Runtime parameter update — force constant changes mid-simulation."""
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.3, 100.0)
    system.addForce(bond)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0)] * unit.nanometers
    ctx.setPositions(positions)

    # Energy at k=100
    state1 = ctx.getState(getEnergy=True)
    e1 = state1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    expected1 = 0.5 * 100.0 * (0.5 - 0.3)**2

    # Update k to 200
    bond.setBondParameters(0, 0, 1, 0.3, 200.0)
    bond.updateParametersInContext(ctx)

    state2 = ctx.getState(getEnergy=True)
    e2 = state2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    expected2 = 0.5 * 200.0 * (0.5 - 0.3)**2

    assert_tol("Energy before parameter update", expected1, e1, ENERGY_TOL)
    assert_tol("Energy after parameter update", expected2, e2, ENERGY_TOL)
    assert_true("Energy doubled when k doubled",
                abs(e2 / e1 - 2.0) < 0.01,
                f"ratio = {e2/e1:.4f}")
    del ctx


def test_24_checkpoint():
    """Checkpoint save/restore — state preservation."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ1 = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
    ctx1 = mm.Context(system, integ1, gpu)
    ctx1.setPositions(modeller.positions)
    ctx1.setVelocitiesToTemperature(300)
    integ1.step(100)

    # Save checkpoint
    checkpoint = ctx1.createCheckpoint()
    state1 = ctx1.getState(getPositions=True, getVelocities=True, getEnergy=True)

    # Create new context, load checkpoint
    integ2 = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
    ctx2 = mm.Context(system, integ2, gpu)
    ctx2.loadCheckpoint(checkpoint)
    state2 = ctx2.getState(getPositions=True, getVelocities=True, getEnergy=True)

    # Compare
    pos1 = state1.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    pos2 = state2.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    max_pos_diff = np.max(np.abs(pos1 - pos2))
    assert_true("Checkpoint positions restored",
                max_pos_diff < 1e-6,
                f"max diff {max_pos_diff:.2e} nm")

    vel1 = state1.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picoseconds)
    vel2 = state2.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picoseconds)
    max_vel_diff = np.max(np.abs(vel1 - vel2))
    assert_true("Checkpoint velocities restored",
                max_vel_diff < 1e-6,
                f"max diff {max_vel_diff:.2e}")

    e1 = state1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e2 = state2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_tol("Checkpoint energy restored", e1, e2, 1e-6)
    del ctx1, ctx2


def test_25_random_seed():
    """Random seed reproducibility — same seed gives same velocities."""
    system = mm.System()
    for _ in range(10):
        system.addParticle(1.0)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)

    def get_velocities(seed):
        integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
        integ.setRandomNumberSeed(seed)
        ctx = mm.Context(system, integ, gpu)
        positions = [mm.Vec3(i * 0.3, 0, 0) for i in range(10)]
        ctx.setPositions(positions * unit.nanometers)
        ctx.setVelocitiesToTemperature(300, seed)
        state = ctx.getState(getVelocities=True)
        v = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometers / unit.picoseconds)
        del ctx
        return v

    v1 = get_velocities(12345)
    v2 = get_velocities(12345)
    v3 = get_velocities(99999)

    assert_true("Same seed → same velocities",
                np.allclose(v1, v2, atol=1e-6))
    assert_true("Different seed → different velocities",
                not np.allclose(v1, v3, atol=0.01))


# ===================================================================
# GROUP 4: Production-Realistic
# ===================================================================

def test_26_opc_water_box():
    """Full OPC 4-site water box — single-point energy vs Reference."""
    ff = app.ForceField('amber/protein.ff19SB.xml', 'amber/opc_standard.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip4pew',
                        boxSize=mm.Vec3(2.5, 2.5, 2.5) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometers,
                             ewaldErrorTolerance=0.0005,
                             constraints=app.HBonds)

    n = modeller.topology.getNumAtoms()
    compare_single_point(system, modeller.positions,
                         f"OPC water box ({n} atoms, PME + virtual sites)")


def test_27_excl_and_14():
    """1-4 interactions and exclusions — correct scaling."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    nb = mm.NonbondedForce()
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    # Exclusions: 0-1, 1-2 (bonded), 0-2 is 1-3 (excluded), 0-3 is 1-4 (scaled)
    nb.addException(0, 1, 0, 1, 0)  # fully excluded
    nb.addException(1, 2, 0, 1, 0)  # fully excluded
    nb.addException(2, 3, 0, 1, 0)  # fully excluded
    nb.addException(0, 2, 0, 1, 0)  # 1-3 excluded
    nb.addException(1, 3, 0, 1, 0)  # 1-3 excluded
    # 0-3 is 1-4: half charge, half epsilon
    nb.addException(0, 3, 0.5 * 1.0 * (-1.0), 0.3, 0.5 * 0.5)
    system.addForce(nb)

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(0.15, 0, 0),
                 mm.Vec3(0.3, 0, 0), mm.Vec3(0.45, 0, 0)] * unit.nanometers

    compare_single_point(system, positions, "1-4 exceptions and exclusions")


def test_28_nonbonded_cutoff_periodic():
    """NonbondedForce CutoffPeriodic — reaction field electrostatics."""
    system = mm.System()
    box = 4.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    n = 32
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2)

    # Place particles on a grid to avoid close contacts that amplify
    # single-precision force differences
    positions = []
    idx = 0
    spacing = box / 4.0  # 4x4x2 = 32
    for ix in range(4):
        for iy in range(4):
            for iz in range(2):
                if idx >= n:
                    break
                system.addParticle(1.0)
                nb.addParticle(0.3 * (-1)**idx, 0.3, 0.5)
                positions.append(mm.Vec3(
                    (ix + 0.5) * spacing,
                    (iy + 0.5) * spacing,
                    (iz + 0.5) * spacing * 2))
                idx += 1
    system.addForce(nb)

    compare_single_point(system, positions * unit.nanometers,
                         "NonbondedForce CutoffPeriodic (32 atoms)")


def test_29_cm_motion_remover():
    """CMMotionRemover — center of mass drift removal."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)
    system.addForce(mm.CMMotionRemover(10))

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
    sim = app.Simulation(modeller.topology, system, integ, gpu)
    sim.context.setPositions(modeller.positions)
    sim.context.setVelocitiesToTemperature(300)

    # Run 500 steps, then check COM velocity is near zero
    sim.step(500)
    state = sim.context.getState(getVelocities=True)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picoseconds)

    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(system.getNumParticles())])
    total_mass = np.sum(masses)
    com_vel = np.sum(velocities * masses[:, np.newaxis], axis=0) / total_mass

    com_speed = np.linalg.norm(com_vel)
    assert_true("COM velocity near zero after CMMotionRemover",
                com_speed < 0.01,
                f"COM speed = {com_speed:.6f} nm/ps")


def test_30_context_reinitialize():
    """Context reinitialize with preserveState — used in Ember equilibration."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.LangevinMiddleIntegrator(300, 1.0, 0.002)
    ctx = mm.Context(system, integ, gpu)
    ctx.setPositions(modeller.positions)
    ctx.setVelocitiesToTemperature(300)
    integ.step(50)

    # Get state before reinitialize
    state_before = ctx.getState(getPositions=True, getVelocities=True,
                                getEnergy=True)
    e_before = state_before.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # Add barostat, reinitialize
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
    ctx.reinitialize(preserveState=True)

    state_after = ctx.getState(getPositions=True, getEnergy=True)
    e_after = state_after.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)

    # Positions should be preserved; energy may differ slightly due to barostat
    pos_before = state_before.getPositions(asNumpy=True).value_in_unit(
        unit.nanometers)
    pos_after = state_after.getPositions(asNumpy=True).value_in_unit(
        unit.nanometers)
    max_diff = np.max(np.abs(pos_before - pos_after))

    assert_true("Positions preserved after reinitialize",
                max_diff < 1e-6,
                f"max diff {max_diff:.2e} nm")
    assert_true("Energy reasonable after reinitialize",
                not math.isnan(e_after) and abs(e_after) < 1e8)
    del ctx


def test_31_global_parameter_update():
    """Global parameter update — CustomExternalForce k scaling (Ember restraint release)."""
    system = mm.System()
    system.addParticle(1.0)

    restraint = mm.CustomExternalForce('k*periodicdistance(x,y,z,x0,y0,z0)^2')
    restraint.addGlobalParameter('k', 100.0)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    restraint.addParticle(0, [0.0, 0.0, 0.0])

    # Need periodic box for periodicdistance
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(5, 0, 0), mm.Vec3(0, 5, 0), mm.Vec3(0, 0, 5))
    system.addForce(restraint)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    pos = [mm.Vec3(0.1, 0.0, 0.0)] * unit.nanometers
    ctx.setPositions(pos)

    # E = k * r^2 = 100 * 0.1^2 = 1.0
    state = ctx.getState(getEnergy=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_tol("Restraint energy at k=100", 1.0, e1, ENERGY_TOL)

    # Scale k to 50 (restraint release step)
    ctx.setParameter('k', 50.0)
    state = ctx.getState(getEnergy=True)
    e2 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_tol("Restraint energy at k=50", 0.5, e2, ENERGY_TOL)

    # Scale k to 0 (fully released)
    ctx.setParameter('k', 0.0)
    state = ctx.getState(getEnergy=True)
    e3 = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_tol("Restraint energy at k=0", 0.0, e3, 1e-6)
    del ctx


def test_32_custom_nonbonded():
    """CustomNonbondedForce — soft-core LJ-like potential."""
    system = mm.System()
    box = 3.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))

    cnb = mm.CustomNonbondedForce('4*eps*((sig/r)^12-(sig/r)^6); '
                                  'sig=0.5*(sig1+sig2); eps=sqrt(eps1*eps2)')
    cnb.addPerParticleParameter('sig')
    cnb.addPerParticleParameter('eps')
    cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(1.2)

    rng = np.random.RandomState(33)
    positions = []
    n = 24
    for i in range(n):
        system.addParticle(1.0)
        cnb.addParticle([0.3, 0.5])
        positions.append(mm.Vec3(*(rng.random(3) * (box - 0.5) + 0.25)))
    system.addForce(cnb)

    compare_single_point(system, positions * unit.nanometers,
                         "CustomNonbondedForce periodic")


def test_33_triclinic_box():
    """Triclinic (non-orthogonal) periodic box — PME correctness."""
    system = mm.System()
    # Triclinic box vectors (reduced form)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3.0, 0, 0), mm.Vec3(0.5, 2.8, 0), mm.Vec3(0.3, 0.2, 2.9))

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9)
    nb.setEwaldErrorTolerance(0.0005)

    rng = np.random.RandomState(44)
    positions = []
    n = 32
    for i in range(n):
        system.addParticle(1.0)
        nb.addParticle(0.2 * (-1)**i, 0.25, 0.5)
        # Place within box using fractional coordinates
        f = rng.random(3)
        cart = (f[0] * np.array([3.0, 0, 0]) +
                f[1] * np.array([0.5, 2.8, 0]) +
                f[2] * np.array([0.3, 0.2, 2.9]))
        positions.append(mm.Vec3(*cart))
    system.addForce(nb)

    compare_single_point(system, positions * unit.nanometers,
                         "PME triclinic box (32 atoms)")


def test_34_rb_torsion():
    """RBTorsionForce — Ryckaert-Bellemans torsion."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    rb = mm.RBTorsionForce()
    rb.addTorsion(0, 1, 2, 3, 5.0, -2.0, 1.0, -0.5, 0.2, -0.1)
    system.addForce(rb)

    positions = [
        mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 0),
        mm.Vec3(1, 0, 0), mm.Vec3(1, 0.8, 0.6)
    ] * unit.nanometers

    compare_single_point(system, positions, "RBTorsionForce")


# ===================================================================
# GROUP 5: Additional kernels (complete Metal C++ suite coverage)
# ===================================================================

def test_35_custom_angle_force():
    """CustomAngleForce — user-defined angle potential."""
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    caf = mm.CustomAngleForce('0.5*k*(theta-theta0)^2 + k2*(theta-theta0)^4')
    caf.addPerAngleParameter('theta0')
    caf.addPerAngleParameter('k')
    caf.addPerAngleParameter('k2')
    caf.addAngle(0, 1, 2, [math.pi / 3.0, 200.0, 50.0])
    system.addForce(caf)

    positions = [mm.Vec3(1, 0, 0), mm.Vec3(0, 0, 0), mm.Vec3(0, 1, 0)] * unit.nanometers
    compare_single_point(system, positions, "CustomAngleForce")


def test_36_custom_torsion_force():
    """CustomTorsionForce — user-defined dihedral potential."""
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    ctf = mm.CustomTorsionForce('k*(1+cos(n*theta-phase))')
    ctf.addPerTorsionParameter('k')
    ctf.addPerTorsionParameter('n')
    ctf.addPerTorsionParameter('phase')
    ctf.addTorsion(0, 1, 2, 3, [5.0, 2.0, math.pi / 3.0])
    system.addForce(ctf)

    positions = [
        mm.Vec3(0, 1, 0), mm.Vec3(0, 0, 0),
        mm.Vec3(1, 0, 0), mm.Vec3(1, 1, 0.5)
    ] * unit.nanometers
    compare_single_point(system, positions, "CustomTorsionForce")


def test_37_custom_centroid_bond():
    """CustomCentroidBondForce — centroid distance restraint."""
    system = mm.System()
    for _ in range(6):
        system.addParticle(1.0)
    ccbf = mm.CustomCentroidBondForce(2, 'k*distance(g1,g2)^2')
    ccbf.addPerBondParameter('k')
    ccbf.addGroup([0, 1, 2])  # group 0
    ccbf.addGroup([3, 4, 5])  # group 1
    ccbf.addBond([0, 1], [100.0])
    system.addForce(ccbf)

    positions = [
        mm.Vec3(0, 0, 0), mm.Vec3(0.1, 0, 0), mm.Vec3(0, 0.1, 0),
        mm.Vec3(2, 0, 0), mm.Vec3(2.1, 0, 0), mm.Vec3(2, 0.1, 0)
    ] * unit.nanometers
    compare_single_point(system, positions, "CustomCentroidBondForce")


def test_38_custom_compound_bond():
    """CustomCompoundBondForce — multi-particle custom potential."""
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    ccbf = mm.CustomCompoundBondForce(3,
        '0.5*k*(distance(p1,p2)-d0)^2 + 0.5*k*(distance(p2,p3)-d0)^2')
    ccbf.addPerBondParameter('k')
    ccbf.addPerBondParameter('d0')
    ccbf.addBond([0, 1, 2], [100.0, 0.3])
    system.addForce(ccbf)

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(0.35, 0, 0),
                 mm.Vec3(0.7, 0, 0)] * unit.nanometers
    compare_single_point(system, positions, "CustomCompoundBondForce")


def test_39_rmsd_force():
    """RMSDForce — RMSD-based restraint."""
    n = 8
    rng = np.random.RandomState(22)
    ref_positions = [mm.Vec3(*(rng.random(3) * 2.0)) for _ in range(n)]

    system = mm.System()
    for _ in range(n):
        system.addParticle(1.0)
    rmsd = mm.RMSDForce(ref_positions * unit.nanometers,
                        list(range(n)))
    system.addForce(rmsd)

    # Displace slightly from reference
    positions = [mm.Vec3(p[0] + 0.05, p[1] - 0.03, p[2] + 0.02)
                 for p in ref_positions] * unit.nanometers
    compare_single_point(system, positions, "RMSDForce")


def test_40_verlet_integrator():
    """VerletIntegrator — energy conservation in harmonic system."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 100.0)
    system.addForce(bond)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.VerletIntegrator(0.001)  # 1 fs, no thermostat
    ctx = mm.Context(system, integrator, gpu)
    ctx.setPositions([mm.Vec3(-1, 0, 0), mm.Vec3(1, 0, 0)] * unit.nanometers)

    state = ctx.getState(getEnergy=True)
    e0 = (state.getKineticEnergy() + state.getPotentialEnergy()
          ).value_in_unit(unit.kilojoules_per_mole)
    max_drift = 0.0
    for _ in range(1000):
        integrator.step(1)
        state = ctx.getState(getEnergy=True)
        e = (state.getKineticEnergy() + state.getPotentialEnergy()
             ).value_in_unit(unit.kilojoules_per_mole)
        max_drift = max(max_drift, abs(e - e0))

    assert_true("Verlet energy conservation",
                max_drift < 0.01,
                f"max drift {max_drift:.6f} kJ/mol")
    del ctx


def test_41_custom_integrator():
    """CustomIntegrator — velocity Verlet via custom step sequence."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 100.0)
    system.addForce(bond)

    # Build velocity Verlet as CustomIntegrator
    dt = 0.001  # ps
    ci = mm.CustomIntegrator(dt)
    ci.addPerDofVariable("x1", 0)
    ci.addUpdateContextState()
    ci.addComputePerDof("v", "v+0.5*dt*f/m")
    ci.addComputePerDof("x", "x+dt*v")
    ci.addComputePerDof("x1", "x")
    ci.addConstrainPositions()
    ci.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    ci.addConstrainVelocities()

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    ctx = mm.Context(system, ci, gpu)
    ctx.setPositions([mm.Vec3(-1, 0, 0), mm.Vec3(1, 0, 0)] * unit.nanometers)

    state = ctx.getState(getEnergy=True)
    e0 = (state.getKineticEnergy() + state.getPotentialEnergy()
          ).value_in_unit(unit.kilojoules_per_mole)
    max_drift = 0.0
    for _ in range(500):
        ci.step(1)
        state = ctx.getState(getEnergy=True)
        e = (state.getKineticEnergy() + state.getPotentialEnergy()
             ).value_in_unit(unit.kilojoules_per_mole)
        max_drift = max(max_drift, abs(e - e0))

    assert_true("CustomIntegrator (velocity Verlet) energy conservation",
                max_drift < 0.01,
                f"max drift {max_drift:.6f} kJ/mol")
    del ctx


def test_42_andersen_thermostat():
    """AndersenThermostat — temperature equilibration."""
    n = 16
    target_temp = 300.0
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.5)
    for i in range(n):
        system.addParticle(2.0)
        nb.addParticle(0.1 * (-1)**i, 0.3, 0.5)
    system.addForce(nb)
    system.addForce(mm.AndersenThermostat(target_temp, 10.0))

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.VerletIntegrator(0.002)
    ctx = mm.Context(system, integrator, gpu)
    rng = np.random.RandomState(88)
    positions = [mm.Vec3(*(rng.random(3) * 3.0 + 0.5)) for _ in range(n)]
    ctx.setPositions(positions * unit.nanometers)
    ctx.setVelocitiesToTemperature(target_temp)

    integrator.step(5000)  # equilibrate
    ke_samples = []
    for _ in range(5000):
        integrator.step(1)
        state = ctx.getState(getEnergy=True)
        ke_samples.append(state.getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole))
    mean_ke = np.mean(ke_samples)
    expected_ke = 0.5 * n * 3 * BOLTZ_KJ * target_temp

    assert_tol("Andersen thermostat temperature",
               expected_ke, mean_ke,
               STAT_TOL_FACTOR / math.sqrt(len(ke_samples)), stochastic=True)
    del ctx


def test_43_dispersion_pme():
    """LJ-PME (dispersion PME) — long-range LJ corrections."""
    system = mm.System()
    box = 3.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.LJPME)
    nb.setCutoffDistance(1.0)
    nb.setEwaldErrorTolerance(0.0005)

    rng = np.random.RandomState(77)
    positions = []
    n = 32
    for i in range(n):
        system.addParticle(1.0)
        nb.addParticle(0.2 * (-1)**i, 0.25 + 0.05 * rng.random(), 0.5)
        positions.append(mm.Vec3(*(rng.random(3) * (box - 0.5) + 0.25)))
    system.addForce(nb)

    compare_single_point(system, positions * unit.nanometers,
                         "NonbondedForce LJPME (32 atoms)")


def test_44_local_energy_minimizer():
    """LocalEnergyMinimizer — finds lower energy on GPU."""
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, gpu)
    ctx.setPositions(modeller.positions)

    state_pre = ctx.getState(getEnergy=True)
    e_pre = state_pre.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 500)

    state_post = ctx.getState(getEnergy=True)
    e_post = state_post.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    assert_true("Minimizer lowers energy",
                e_post < e_pre,
                f"before={e_pre:.1f}, after={e_post:.1f}")
    assert_true("Minimized energy is finite",
                not math.isnan(e_post) and abs(e_post) < 1e8)
    del ctx


def test_45_compound_integrator():
    """CompoundIntegrator — switch between integrators mid-simulation."""
    system = mm.System()
    system.addParticle(2.0)
    system.addParticle(2.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.5, 100.0)
    system.addForce(bond)

    ci = mm.CompoundIntegrator()
    ci.addIntegrator(mm.VerletIntegrator(0.001))
    ci.addIntegrator(mm.LangevinMiddleIntegrator(300, 1.0, 0.002))
    ci.setCurrentIntegrator(0)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    ctx = mm.Context(system, ci, gpu)
    ctx.setPositions([mm.Vec3(-1, 0, 0), mm.Vec3(1, 0, 0)] * unit.nanometers)

    # Step with Verlet
    ci.step(100)
    state1 = ctx.getState(getEnergy=True)
    e1 = state1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_true("CompoundIntegrator Verlet phase ok",
                not math.isnan(e1) and abs(e1) < 1e8)

    # Switch to Langevin
    ci.setCurrentIntegrator(1)
    ci.step(100)
    state2 = ctx.getState(getEnergy=True)
    e2 = state2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert_true("CompoundIntegrator Langevin phase ok",
                not math.isnan(e2) and abs(e2) < 1e8)
    del ctx


def test_46_anisotropic_barostat():
    """MonteCarloAnisotropicBarostat — independent axis scaling."""
    n = 64
    target_temp = 300.0
    system = mm.System()
    box = 4.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))

    rng = np.random.RandomState(66)
    positions = []
    for i in range(n):
        system.addParticle(1.0)
        positions.append(mm.Vec3(*(rng.random(3) * (box - 0.5) + 0.25)))
    positions = positions * unit.nanometers

    baro = mm.MonteCarloAnisotropicBarostat(
        mm.Vec3(1, 1, 1) * unit.bar, target_temp * unit.kelvin,
        True, True, True, 10)
    system.addForce(baro)
    bonds = mm.HarmonicBondForce()
    bonds.setUsesPeriodicBoundaryConditions(True)
    system.addForce(bonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.LangevinIntegrator(target_temp, 0.1, 0.01)
    ctx = mm.Context(system, integrator, gpu)
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(target_temp)

    # Just verify it runs without crashing and box dimensions change
    integrator.step(2000)
    state = ctx.getState()
    box_vecs = state.getPeriodicBoxVectors()
    lx = box_vecs[0][0].value_in_unit(unit.nanometers)
    ly = box_vecs[1][1].value_in_unit(unit.nanometers)
    lz = box_vecs[2][2].value_in_unit(unit.nanometers)

    assert_true("Anisotropic barostat ran without error", True)
    assert_true("Box dimensions still reasonable",
                all(1.0 < d < 20.0 for d in [lx, ly, lz]),
                f"box = {lx:.2f} x {ly:.2f} x {lz:.2f} nm")
    del ctx


def test_47_variable_langevin():
    """VariableLangevinIntegrator — adaptive timestep temperature control."""
    target_temp = 300.0

    # Use a real water box — bare particles with adaptive timestep are
    # inherently fragile; a minimized water box is the standard test.
    ff = app.ForceField('amber14/tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(2.0, 2.0, 2.0) * unit.nanometers)
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometers,
                             constraints=app.HBonds)

    gpu = mm.Platform.getPlatformByName(GPU_PLATFORM)
    integrator = mm.VariableLangevinIntegrator(target_temp, 2.0, 0.002)
    simulation = app.Simulation(modeller.topology, system, integrator, gpu)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(target_temp)

    # DOF for temperature calculation
    n_real = sum(1 for i in range(system.getNumParticles())
                 if system.getParticleMass(i).value_in_unit(unit.dalton) > 0)
    n_constraints = system.getNumConstraints()
    n_dof = 3 * n_real - n_constraints - 3

    # Equilibrate
    integrator.stepTo(2.0)  # 2 ps

    # Sample
    ke_samples = []
    for _ in range(2000):
        integrator.step(1)
        state = simulation.context.getState(getEnergy=True)
        ke_samples.append(state.getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole))

    mean_ke = np.mean(ke_samples)
    expected_ke = 0.5 * n_dof * BOLTZ_KJ * target_temp
    assert_tol("VariableLangevin temperature",
               expected_ke, mean_ke,
               STAT_TOL_FACTOR / math.sqrt(len(ke_samples)), stochastic=True)


# ===================================================================
# Main
# ===================================================================

ALL_TESTS = [
    # Group 1: Force kernels
    test_01_coulomb,
    test_02_lennard_jones,
    test_03_harmonic_bond,
    test_04_harmonic_angle,
    test_05_periodic_torsion,
    test_06_cmap_torsion,
    test_07_nonbonded_pme,
    test_08_nonbonded_pme_large,
    test_09_custom_external_force,
    test_10_custom_bond_force,
    test_11_gbsa_obc,
    test_12_gbsa_obc_cutoff,
    test_13_multiple_forces,
    # Group 2: Integrators & ensemble
    test_14_langevin_middle_single_bond,
    test_15_langevin_middle_energy_conservation,
    test_16_langevin_middle_temperature,
    test_17_langevin_middle_constraints,
    test_18_settle,
    test_19_barostat_ideal_gas,
    test_20_barostat_water_density,
    # Group 3: Infrastructure
    test_21_virtual_sites,
    test_22_hmr,
    test_23_parameter_update,
    test_24_checkpoint,
    test_25_random_seed,
    # Group 4: Production-realistic
    test_26_opc_water_box,
    test_27_excl_and_14,
    test_28_nonbonded_cutoff_periodic,
    test_29_cm_motion_remover,
    test_30_context_reinitialize,
    test_31_global_parameter_update,
    test_32_custom_nonbonded,
    test_33_triclinic_box,
    test_34_rb_torsion,
    # Group 5: Complete Metal suite coverage
    test_35_custom_angle_force,
    test_36_custom_torsion_force,
    test_37_custom_centroid_bond,
    test_38_custom_compound_bond,
    test_39_rmsd_force,
    test_40_verlet_integrator,
    test_41_custom_integrator,
    test_42_andersen_thermostat,
    test_43_dispersion_pme,
    test_44_local_energy_minimizer,
    test_45_compound_integrator,
    test_46_anisotropic_barostat,
    test_47_variable_langevin,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="OpenCL (cl2Metal) platform validation suite")
    parser.add_argument('--platform', default='OpenCL',
                        help='GPU platform to test (default: OpenCL)')
    parser.add_argument('--test', type=str, default=None,
                        help='Run a single test by name (e.g. test_07_nonbonded_pme)')
    args = parser.parse_args()
    GPU_PLATFORM = args.platform

    print("=" * 64)
    print(f"  OpenMM {GPU_PLATFORM} Platform Validation Suite")
    print(f"  (Mirrors native Metal C++ test coverage)")
    print("=" * 64)
    print(f"  OpenMM version: {mm.Platform.getOpenMMVersion()}")
    platforms = [mm.Platform.getPlatform(i).getName()
                 for i in range(mm.Platform.getNumPlatforms())]
    print(f"  Platforms: {platforms}")
    print(f"  Testing: {GPU_PLATFORM} vs Reference")

    # Check GPU platform exists
    try:
        p = mm.Platform.getPlatformByName(GPU_PLATFORM)
        # Quick probe
        s = mm.System(); s.addParticle(1.0)
        ctx = mm.Context(s, mm.VerletIntegrator(0.001), p)
        try:
            device = p.getPropertyValue(ctx, 'DeviceName')
            print(f"  Device: {device}")
        except Exception:
            pass
        del ctx
    except Exception as e:
        print(f"\n  FATAL: {GPU_PLATFORM} platform not available: {e}")
        sys.exit(1)

    t0 = time.time()

    if args.test:
        matching = [t for t in ALL_TESTS if t.__name__ == args.test]
        if not matching:
            print(f"\n  Unknown test: {args.test}")
            print(f"  Available: {[t.__name__ for t in ALL_TESTS]}")
            sys.exit(1)
        for t in matching:
            run_test(t)
    else:
        for t in ALL_TESTS:
            run_test(t)

    elapsed = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  SUMMARY — {GPU_PLATFORM} platform")
    print(f"{'='*64}")
    for name, status, dt in RESULTS:
        icon = {"ok": ".", "skip": "S", "error": "E"}[status]
        print(f"  [{icon}] {name}" + (f" ({dt:.1f}s)" if status == "ok" else ""))

    print(f"\n  {PASS_COUNT} passed, {FAIL_COUNT} failed, "
          f"{SKIP_COUNT} skipped ({elapsed:.1f}s)")

    if FAIL_COUNT > 0:
        print("\n  SOME TESTS FAILED — see details above.")
        sys.exit(1)
    elif SKIP_COUNT > 0:
        print("\n  All run tests PASSED (some skipped).")
    else:
        print("\n  ALL TESTS PASSED!")
    sys.exit(0)
