from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.linalg import solve
from numba import njit

AtomName = 'H'

def fal_height_upsampler():
    atmos = Falc82()
    atmos.convert_scales()

    def resample(factor, outer=False):
        nonlocal atmos
        NspaceOriginal = atmos.Nspace
        Nspace = factor * (NspaceOriginal - 1) + 1
        if outer:
            Nspace += 1
        originalGrid = np.linspace(0, 1, NspaceOriginal)
        newGrid = np.linspace(0, 1, Nspace)

        height = interp1d(originalGrid, atmos.height, kind=3)(newGrid)
        temp = interp1d(originalGrid, atmos.temperature, kind=3)(newGrid)
        ne = interp1d(originalGrid, atmos.ne, kind=3)(newGrid)
        vlos = interp1d(originalGrid, atmos.vlos, kind=3)(newGrid)
        vturb = interp1d(originalGrid, atmos.vturb, kind=3)(newGrid)
        nHTot = interp1d(originalGrid, atmos.nHTot, kind=3)(newGrid)

        newAtmos = lw.Atmosphere(lw.ScaleType.Geometric, depthScale=height, temperature=temp, 
                                 ne=ne, vlos=vlos, vturb=vturb, nHTot=nHTot)
        newAtmos.height = newAtmos.depthScale
        # Sue me.
        newAtmos.tau_ref = np.ones_like(newAtmos.depthScale)
        newAtmos.cmass = np.ones_like(newAtmos.depthScale)

        return newAtmos

    return resample

def residual(ctx):
    """
    Residual for the statistical equilibrium equations.
    """
    Nspace = ctx.atmos.Nspace
    atom = ctx.activeAtoms[0]
    Nlevel = atom.Nlevel
    resid = np.zeros((Nlevel, Nspace))

    for k in range(Nspace):
        iEliminate = -1
        Gamma = np.copy(atom.Gamma[:, :, k])

        Gamma[iEliminate, :] = 1.0

        f = np.zeros(Nlevel)
        f[iEliminate] = atom.nTotal[k]

        resid[:, k] = f - Gamma @ atom.n[:, k]

    return resid

def stat_equil(ctx):
    """Update the populations of all active species towards statistical
    equilibrium, using the current version of the Gamma matrix.
    Adapted from version in Lightspinner.
    Returns
    -------
    maxRelChange : float
        The maximum relative change in any of the atomic populations (at
        the depth point with maximum population change).
    """
    Nspace = ctx.atmos.Nspace

    maxRelChange = 0.0
    for atom in ctx.activeAtoms:
        maxRelChange = max(maxRelChange, stat_equil_impl(atom.Gamma, atom.nTotal, atom.n))

    return maxRelChange

@njit(cache=True)
def stat_equil_impl(Gamma, nTotal, n):
    maxRelChange = 0.0
    Nlevel = Gamma.shape[0]
    Nspace = Gamma.shape[2]
    Gam = np.zeros((Nlevel, Nlevel))
    f = np.zeros(Nlevel)
    nOld = np.zeros(Nlevel)

    for k in range(Nspace):
        # NOTE(cmo): Find the level with the maximum population at this depth point
        iEliminate = -1
        # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
        Gam[...] = np.copy(Gamma[:, :, k])

        # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
        Gam[iEliminate, :] = 1.0
        # NOTE(cmo): Set solution vector to 0 (as per stat. eq.) other than entry for which we are conserving population
        f[:] = 0.0
        f[iEliminate] = nTotal[k]

        # NOTE(cmo): Solve Gamma . n = 0 (constrained by conservation equation)
        nOld[:] = n[:, k]
        nNew = np.linalg.solve(Gam, f)
        # NOTE(cmo): Compute relative change and update populations
        change = np.abs(1.0 - nOld / nNew)
        maxRelChange = max(maxRelChange, change.max())
        n[:, k] = nNew

    return maxRelChange

def stat_equil_rhs(ctx, rhs):
    """
    Solve the kinetic equilibrium equations for a given RHS.
    """
    atom = ctx.activeAtoms[0]
    return stat_equil_rhs_impl(atom.Gamma, rhs, atom.n)


@njit(cache=True)
def stat_equil_rhs_impl(Gamma, rhs, n):
    maxRelChange = 0.0
    Nlevel = Gamma.shape[0]
    Nspace = Gamma.shape[2]
    Gam = np.zeros((Nlevel, Nlevel))
    nOld = np.zeros(Nlevel)
    rhsk = np.zeros(Nlevel)

    for k in range(Nspace):
        # NOTE(cmo): Find the level with the maximum population at this depth point
        iEliminate = -1
        # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
        Gam[...] = Gamma[:, :, k]

        # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
        Gam[iEliminate, :] = 1.0

        # NOTE(cmo): Solve Gamma . n = rhs (constrained by conservation equation)
        nOld[:] = n[:, k]
        rhsk[:] = rhs[:, k]
        nNew = np.linalg.solve(Gam, rhsk)
        # NOTE(cmo): Compute relative change and update populations
        change = np.abs(1.0 - nOld / nNew)
        maxRelChange = max(maxRelChange, change.max())
        n[:, k] = nNew

    return maxRelChange


def prolong(coarse, fine):
    assert coarse.shape[0] == fine.shape[0]
    assert coarse.shape[1] * 2 - 1 == fine.shape[1]

    fine[:, ::2] = coarse
    fine[:, 3:-3:2] = -1/16 * coarse[:, :-3] + 9/16 * coarse[:, 1:-2] + 9/16 * coarse[:, 2:-1] - 1/16 * coarse[:, 3:]
    fine[:, 1] = 3/8 * coarse[:, 0] + 3/4 * coarse[:, 1] - 1/8 * coarse[:, 2]
    fine[:, -2] = 3/8 * coarse[:, -1] + 3/4 * coarse[:, -2] - 1/8 * coarse[:, -3]

def restrict(fine, coarse):
    assert coarse.shape[0] == fine.shape[0]
    assert coarse.shape[1] * 2 - 1 == fine.shape[1]

    coarse[:, 1:-1] = 0.25 * (fine[:, 1:-3:2] + 2 * fine[:, 2:-2:2] + fine[:, 3:-1:2])
    coarse[:, 0] = fine[:, 0]
    coarse[:, -1] = fine[:, -1]

# @profile
def v_cycle(ctx, eqPops, gridIdx, rhs):
    global nIter
    if gridIdx == 0:
        ctx[gridIdx].formal_sol_gamma_matrices()
        initialError = stat_equil_rhs(ctx[gridIdx], rhs)
        print('Initial: %e' % initialError)
        error = initialError
        while error > 0.1 * initialError:
            dJ = ctx[gridIdx].formal_sol_gamma_matrices()
            error = stat_equil_rhs(ctx[gridIdx], rhs)
            nIter[gridIdx] += 1
            print('Initial: %e, now: %e' % (initialError, error))
        # print(nIter)
        Rc = error
    else:
        # pre-smooth
        for nu in range(nu1):
            dJ = ctx[gridIdx].formal_sol_gamma_matrices()
            delta = stat_equil_rhs(ctx[gridIdx], rhs)
            nIter[gridIdx] += 1

        ctx[gridIdx].formal_sol_gamma_matrices(lambdaIterate=True)
        nIter[gridIdx] += 1
        # Compute residual
        fineResidual = residual(ctx[gridIdx])
        # Restrict
        coarseResidual = np.zeros_like(eqPops[gridIdx-1][AtomName])
        restrict(fineResidual, coarseResidual)
        restrict(eqPops[gridIdx][AtomName], eqPops[gridIdx-1][AtomName])

        # Coarse grid rhs
        nInit = np.copy(eqPops[gridIdx-1][AtomName])
        ctx[gridIdx-1].formal_sol_gamma_matrices(lambdaIterate=True)
        nIter[gridIdx-1] += 1
        coarseRhs2 = np.zeros_like(coarseResidual)
        atom = ctx[gridIdx-1].activeAtoms[0]
        Gamma = np.zeros((atom.Nlevel, atom.Nlevel))
        for k in range(ctx[gridIdx-1].atmos.Nspace):
            Gamma[...] = atom.Gamma[:, :, k]
            Gamma[-1, :] = 1.0
            coarseRhs2[:, k] = Gamma @ atom.n[:, k]
        coarseRhs = coarseRhs2 + coarseResidual

        # Recursively get population update from coarser grids
        v_cycle(ctx, eqPops, gridIdx-1, coarseRhs)

        # Coarse population error
        coarseError = eqPops[gridIdx-1][AtomName] - nInit
        fineError = np.zeros_like(fineResidual)
        prolong(coarseError, fineError)
        eqPops[gridIdx][AtomName][:] += fineError

        # Post smooth
        for nu in range(nu2):
            dJ = ctx[gridIdx].formal_sol_gamma_matrices()
            delta = stat_equil_rhs(ctx[gridIdx], rhs)
            nIter[gridIdx] += 1
        Rc = delta
    return Rc


eta1 = 1e-1
eta2 = 5e-4
nu1 = 2
nu2 = 4

# @profile
# def main():
fal_sampler = fal_height_upsampler()
falHeirarchy = [fal_sampler(2**i) for i in range(4)]
for f in falHeirarchy:
    f.quadrature(5)

aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active(AtomName)
spect = aSet.compute_wavelength_grid()

eqPops = [aSet.compute_eq_pops(atmos) for atmos in falHeirarchy]

ctx = [lw.Context(atmos, spect, pops, ngOptions=lw.NgOptions(0,0,0), conserveCharge=False, initSol=lw.InitialSolution.Lte, Nthreads=8) for atmos, pops in zip(falHeirarchy, eqPops)]


Nested = False
mgStart = time.time()
global nIter
nIter = [0 for f in falHeirarchy]
if Nested:
    # Full multigrid V-cycle
    # NOTE(cmo): Initial guess on coarsest grid
    for i in range(100):
        dJ = ctx[0].formal_sol_gamma_matrices()
        delta = stat_equil(ctx[0])
        nIter[0] += 1

        if delta < eta1:
            break
    else:
        raise lw.ConvergenceError('Coarse not converged')

    for gridIdx in range(1, len(ctx)):
        Rc = 1.0
        while Rc > eta2:
            prolong(eqPops[gridIdx-1][AtomName], eqPops[gridIdx][AtomName])
            rhs = np.zeros_like(eqPops[gridIdx][AtomName])
            rhs[-1, :] = eqPops[gridIdx].atomicPops[AtomName].nTotal
            Rc = v_cycle(ctx, eqPops, gridIdx, rhs)
            print('Rc after cycle %d, %e' % (gridIdx, Rc))
else:
    # V-cycle
    Rc = 1.0
    count = 0
    while Rc > eta2:
        rhs = np.zeros_like(eqPops[-1][AtomName])
        rhs[-1, :] = eqPops[-1].atomicPops[AtomName].nTotal
        Rc = v_cycle(ctx, eqPops, len(ctx)-1, rhs)
        count += 1
        print('Rc after cycle %d, %e' % (count, Rc))
        

mgEnd = time.time()

# Reference MALI solution
trueEqPops = aSet.compute_eq_pops(falHeirarchy[-1])
trueCtx = lw.Context(falHeirarchy[-1], spect, trueEqPops, ngOptions=lw.NgOptions(0,0,0), conserveCharge=False, initSol=lw.InitialSolution.Lte, Nthreads=8)

maliStart = time.time()
for i in range(5000):
    dJ = trueCtx.formal_sol_gamma_matrices()
    delta = stat_equil(trueCtx)
    print(delta)
    if delta < Rc:
        print('True took %d iterations.' % (i+1))
        break
maliEnd = time.time()

mgDuration = mgEnd - mgStart
maliDuration = maliEnd - maliStart
print(mgDuration, maliDuration, mgDuration/maliDuration)
