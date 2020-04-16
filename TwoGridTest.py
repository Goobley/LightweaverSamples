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

def fal_height_upsampler():
    atmos = Falc82()
    atmos.convert_scales()

    def resample(factor, outer=False):
        nonlocal atmos
        NspaceOriginal = atmos.Nspace
        Nspace = NspaceOriginal * factor - 1
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

def iterate_ctx(ctx, atmos, eqPops, prd=True, Nscatter=3, NmaxIter=500, updateLte=False):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if prd:
            dRho = ctx.prd_redistribute(maxIter=5)

        if updateLte:
            eqPops.update_lte_atoms_Hmin_pops(atmos)


        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

def residual(ctx):
    Nspace = ctx.atmos.Nspace
    atom = ctx.activeAtoms[0]
    Nlevel = atom.Nlevel
    resid = np.zeros((Nlevel, Nspace))

    for k in range(Nspace):
        iEliminate = -1
        # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
        Gamma = np.copy(atom.Gamma[:, :, k])

        # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
        Gamma[iEliminate, :] = 1.0

        f = np.zeros(Nlevel)
        f[iEliminate] = atom.nTotal[k]

        resid[:, k] = f - Gamma @ atom.n[:, k]

    return resid

def stat_equil(ctx):
    """Update the populations of all active species towards statistical
    equilibrium, using the current version of the Gamma matrix.
    Returns
    -------
    maxRelChange : float
        The maximum relative change in any of the atomic populations (at
        the depth point with maximum population change).
    """
    Nspace = ctx.atmos.Nspace

    maxRelChange = 0.0
    for atom in ctx.activeAtoms:
        Nlevel = atom.Nlevel
        for k in range(Nspace):
            # NOTE(cmo): Find the level with the maximum population at this depth point
            iEliminate = -1
            # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
            Gamma = np.copy(atom.Gamma[:, :, k])

            # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
            Gamma[iEliminate, :] = 1.0
            # NOTE(cmo): Set solution vector to 0 (as per stat. eq.) other than entry for which we are conserving population
            nk = np.zeros(Nlevel)
            nk[iEliminate] = atom.nTotal[k]

            # NOTE(cmo): Solve Gamma . n = 0 (constrained by conservation equation)
            nOld = np.copy(atom.n[:, k])
            nNew = solve(Gamma, nk)
            # NOTE(cmo): Compute relative change and update populations
            change = np.abs(1.0 - nOld / nNew)
            maxRelChange = max(maxRelChange, change.max())
            atom.n[:, k] = nNew

    return maxRelChange

def stat_equil_rhs(ctx, rhs):
    """Update the populations of all active species towards statistical
    equilibrium, using the current version of the Gamma matrix.
    Returns
    -------
    maxRelChange : float
        The maximum relative change in any of the atomic populations (at
        the depth point with maximum population change).
    """
    Nspace = ctx.atmos.Nspace

    maxRelChange = 0.0
    atom = ctx.activeAtoms[0]
    Nlevel = atom.Nlevel
    for k in range(Nspace):
        # NOTE(cmo): Find the level with the maximum population at this depth point
        iEliminate = -1
        # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
        Gamma = np.copy(atom.Gamma[:, :, k])

        # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
        Gamma[iEliminate, :] = 1.0

        # NOTE(cmo): Solve Gamma . n = 0 (constrained by conservation equation)
        nOld = np.copy(atom.n[:, k])
        nNew = solve(Gamma, rhs[:, k])
        # NOTE(cmo): Compute relative change and update populations
        change = np.abs(1.0 - nOld / nNew)
        maxRelChange = max(maxRelChange, change.max())
        atom.n[:, k] = nNew

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


fal_sampler = fal_height_upsampler()
coarse = fal_sampler(2, outer=True)
fine = fal_sampler(4)
coarse.quadrature(5)
fine.quadrature(5)
aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()

coarseEqPops = aSet.compute_eq_pops(coarse)
fineEqPops = aSet.compute_eq_pops(fine)
trueEqPops = aSet.compute_eq_pops(fine)

coarseCtx = lw.Context(coarse, spect, coarseEqPops, ngOptions=lw.NgOptions(0,0,0), conserveCharge=False, initSol=lw.InitialSolution.Lte, Nthreads=8)
fineCtx = lw.Context(fine, spect, fineEqPops, ngOptions=lw.NgOptions(0,0,0), conserveCharge=False, initSol=lw.InitialSolution.Lte, Nthreads=8)
trueCtx = lw.Context(fine, spect, trueEqPops, ngOptions=lw.NgOptions(0,0,0), conserveCharge=False, initSol=lw.InitialSolution.Lte, Nthreads=8)

# NOTE(cmo): Direct MALI on fineCtx takes 101 iterations to 1e-3, 143 to 1e-4
# Two grid takes 102

eta1 = 5e-2
eta2 = 1e-4
nu1 = 2
nu2 = 4

mgStart = time.time()
nIter = 0
# NOTE(cmo): Initial guess on coarsest grid
for i in range(100):
    dJ = coarseCtx.formal_sol_gamma_matrices()
    delta = stat_equil(coarseCtx)
    nIter += 1

    if delta < eta1:
        break
else:
    raise lw.ConvergenceError('Coarse not converged')

Rc = 1.0
while Rc > eta2:
    prolong(coarseEqPops['Ca'], fineEqPops['Ca'])


    # pre-smooth
    for nu in range(nu1):
        dJ = fineCtx.formal_sol_gamma_matrices()
        delta = stat_equil(fineCtx)
        nIter += 1

    fineCtx.formal_sol_gamma_matrices(lambdaIterate=True)
    nIter += 1
    # Compute residual
    fineResidual = residual(fineCtx)
    # Restrict
    coarseResidual = np.zeros_like(coarseEqPops['Ca'])
    restrict(fineResidual, coarseResidual)
    restrict(fineEqPops['Ca'], coarseEqPops['Ca'])

    # Error on coarse grid
    nInit = np.copy(coarseEqPops['Ca'])

    coarseCtx.formal_sol_gamma_matrices(lambdaIterate=True)
    nIter += 1
    coarseRhs2 = np.zeros_like(coarseResidual)
    atom = coarseCtx.activeAtoms[0]
    Gamma = np.zeros((atom.Nlevel, atom.Nlevel))
    for k in range(coarseCtx.atmos.Nspace):
        Gamma[...] = atom.Gamma[:, :, k]
        Gamma[-1, :] = 1.0
        coarseRhs2[:, k] = Gamma @ atom.n[:, k]
    coarseRhs = coarseRhs2 + coarseResidual

    initialError = stat_equil_rhs(coarseCtx, coarseRhs)
    print('Initial: %e' % initialError)
    error = initialError
    while error > 0.1 * initialError:
        dJ = coarseCtx.formal_sol_gamma_matrices()
        error = stat_equil_rhs(coarseCtx, coarseRhs)
        nIter += 1
        print('Initial: %e, now %e' % (initialError, error))
    # print(nIter)

    coarseError = coarseEqPops['Ca'] - nInit
    fineError = np.zeros_like(fineResidual)
    prolong(coarseError, fineError)
    fineEqPops['Ca'][:] += fineError

    for nu in range(nu2):
        dJ = fineCtx.formal_sol_gamma_matrices()
        delta = stat_equil(fineCtx)
        nIter += 1
    Rc = delta
    print('Rc: %e' % Rc)

mgEnd = time.time()


maliStart = time.time()
for i in range(5000):
    dJ = trueCtx.formal_sol_gamma_matrices()
    delta = stat_equil(trueCtx)
    if delta < eta2:
        print('True took %d iterations.' % (i+1))
        break
maliEnd = time.time()

mgDuration = mgEnd - mgStart
maliDuration = maliEnd - maliStart
print(mgDuration, maliDuration, mgDuration/maliDuration)