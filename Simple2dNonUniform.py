from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, Mg_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
from lightweaver.atmosphere import ZeroRadiation
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
from lightweaver.utils import NgOptions, get_default_molecule_path
from weno4 import weno4
from numba import njit


# NOTE(cmo): These appear to work pretty well on initial inspection. May need
# some kind of limiting for the prolongation case.
def restrict(fine, coarse):
    assert coarse.shape[0] == fine.shape[0]
    assert coarse.shape[1] * 2 - 1 == fine.shape[1]
    assert coarse.shape[2] * 2 - 1 == fine.shape[2]

    coarse[:, 0, :] = fine[:, 0, ::2]
    coarse[:, -1, :] = fine[:, -1, ::2]
    coarse[:, :,  0] = fine[:, ::2,  0]
    coarse[:, :, -1] = fine[:, ::2, -1]
    for j in range(1, coarse.shape[2]-1):
        coarse[:, 1:-1, j] = 1/16 * (2 * fine[:, 1:-3:2, 2*j] + 4 * fine[:, 2:-2:2, 2*j] + 2 * fine[:, 3:-1:2, 2*j]
                                     + fine[:, 1:-3:2, 2*j-1] + 2 * fine[:, 2:-2:2, 2*j-1]   + fine[:, 3:-1:2, 2*j-1]
                                     + fine[:, 1:-3:2, 2*j+1] + 2 * fine[:, 2:-2:2, 2*j+1]   + fine[:, 3:-1:2, 2*j+1])

def prolong_2d_plane(coarseZ, fineZ, coarseX, fineX, coarse, fine):
    assert coarse.shape[0] * 2 - 1 == fine.shape[0]
    assert coarse.shape[1] * 2 - 1 == fine.shape[1]

    for j in range(coarse.shape[1]):
            fine[:, 2 * j] = weno4(fineZ, coarseZ, coarse[:, j])

    for k in range(fine.shape[0]):
            fine[k, :] = weno4(fineX, coarseX, np.copy(fine[k, ::2]))


def prolong(coarseZ, fineZ, coarseX, fineX, coarse, fine):
    assert coarse.shape[0] == fine.shape[0]
    assert coarse.shape[1] * 2 - 1 == fine.shape[1]
    assert coarse.shape[2] * 2 - 1 == fine.shape[2]

    for p in range(fine.shape[0]):
        prolong_2d_plane(coarseZ, fineZ, coarseX, fineX, coarse[p], fine[p])


def upsample_spatial_axis(coarse):
    fine = np.zeros(2 * coarse.shape[0] - 1)
    fine[0] = coarse[0]
    fine[-1] = coarse[-1]
    fine[2:-2:2] = coarse[1:-1]
    fine[1:-1:2] = 0.5 * (coarse[1:] + coarse[:-1])
    return fine

def upsample_atmos_2d(atmos):
    fineX = upsample_spatial_axis(atmos.x)
    fineZ = upsample_spatial_axis(atmos.z)
    fineTemp = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.temperature.reshape(atmos.Nz, atmos.Nx), fineTemp)
    fineVx = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.vx.reshape(atmos.Nz, atmos.Nx), fineVx)
    fineVz = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.vz.reshape(atmos.Nz, atmos.Nx), fineVz)
    fineVturb = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.vturb.reshape(atmos.Nz, atmos.Nx), fineVturb)
    fineNHTot = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.nHTot.reshape(atmos.Nz, atmos.Nx), fineNHTot)
    fineNe = np.zeros((fineZ.shape[0], fineX.shape[0]))
    prolong_2d_plane(atmos.z, fineZ, atmos.x, fineX, atmos.ne.reshape(atmos.Nz, atmos.Nx), fineNe)

    atmosFine = lw.Atmosphere.make_2d(height=fineZ, x=fineX, temperature=fineTemp,
                                      vx=fineVx, vz=fineVz, vturb=fineVturb,
                                      ne=fineNe, nHTot=fineNHTot)
    return atmosFine


def residual(ctx):
    """
    Residual for the statistical equilibrium equations.
    """
    Nspace = ctx.atmos.Nspace
    atoms = ctx.activeAtoms
    resid = [np.zeros((atom.Nlevel, Nspace)) for atom in atoms]

    for i, atom in enumerate(atoms):
        for k in range(Nspace):
            iEliminate = -1
            Gamma = np.copy(atom.Gamma[:, :, k])

            Gamma[iEliminate, :] = 1.0

            f = np.zeros(atom.Nlevel)
            f[iEliminate] = atom.nTotal[k]

            resid[i][:, k] = f - Gamma @ atom.n[:, k]

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
    atoms = ctx.activeAtoms
    maxRelChange = 0.0
    for i, atom in enumerate(atoms):
        maxRelChange = max(maxRelChange, stat_equil_rhs_impl(atom.Gamma, rhs[i], atom.n))

    return maxRelChange


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
        nInitial = [np.copy(eqPops[gridIdx][a.element]) for a in ctx[gridIdx].activeAtoms]
        # pre-smooth
        for nu in range(nu1):
            dJ = ctx[gridIdx].formal_sol_gamma_matrices()
            delta = stat_equil_rhs(ctx[gridIdx], rhs)
            nIter[gridIdx] += 1

        coarse = ctx[gridIdx-1].atmos
        fine = ctx[gridIdx].atmos

        ctx[gridIdx].formal_sol_gamma_matrices(lambdaIterate=True)
        nIter[gridIdx] += 1
        # Compute residual
        fineResidual = residual(ctx[gridIdx])
        # Restrict
        coarseResidual = [np.zeros_like(eqPops[gridIdx-1][a.element]) for a in ctx[gridIdx].activeAtoms]
        for i in range(len(coarseResidual)):
            restrict(fineResidual[i].reshape(-1, fine.Nz, fine.Nx),
                     coarseResidual[i].reshape(-1, coarse.Nz, coarse.Nx))
        for i, a in enumerate(ctx[gridIdx].activeAtoms):
            restrict(eqPops[gridIdx][a.element].reshape(-1, fine.Nz, fine.Nx),
                    eqPops[gridIdx-1][a.element].reshape(-1, coarse.Nz, coarse.Nx))

        # Coarse grid rhs
        nInit = [np.copy(eqPops[gridIdx-1][a.element]) for a in ctx[gridIdx].activeAtoms]
        ctx[gridIdx-1].formal_sol_gamma_matrices(lambdaIterate=True)
        nIter[gridIdx-1] += 1
        coarseRhs2 = [np.zeros_like(c) for c in coarseResidual]
        coarseRhs = []
        for i, atom in enumerate(ctx[gridIdx-1].activeAtoms):
            Gamma = np.zeros((atom.Nlevel, atom.Nlevel))
            for k in range(ctx[gridIdx-1].atmos.Nspace):
                Gamma[...] = atom.Gamma[:, :, k]
                Gamma[-1, :] = 1.0
                coarseRhs2[i][:, k] = Gamma @ atom.n[:, k]
            coarseRhs.append(coarseRhs2[i] + coarseResidual[i])

        # Recursively get population update from coarser grids
        v_cycle(ctx, eqPops, gridIdx-1, coarseRhs)

        # Coarse population error
        coarseError = [eqPops[gridIdx-1][a.element] - nInit[i] for i, a in enumerate(ctx[gridIdx].activeAtoms)]
        fineError = [np.zeros_like(r) for r in fineResidual]
        for i in range(len(coarseError)):
            prolong(coarse.z, fine.z, coarse.x, fine.x,
                    coarseError[i].reshape(-1, coarse.Nz, coarse.Nx),
                    fineError[i].reshape(-1, fine.Nz, fine.Nx))
        for i, a in enumerate(ctx[gridIdx].activeAtoms):
            eqPops[gridIdx][a.element][:] += fineError[i]

        # Post smooth
        for nu in range(nu2):
            dJ = ctx[gridIdx].formal_sol_gamma_matrices()
            delta = stat_equil_rhs(ctx[gridIdx], rhs)
            nIter[gridIdx] += 1

        changes = [np.max(np.abs(eqPops[gridIdx][a.element] - nInitial[i]) / nInitial[i]) for i, a in enumerate(ctx[gridIdx].activeAtoms)]
        Rc = max(changes)
    return Rc

atmos1d = Falc82()
Nx = 20
Lx = 2e6
x = np.linspace(0, Lx, Nx)
oldZ = np.copy(atmos1d.height)
Nz = 40
z = weno4(np.linspace(0, 1, Nz), np.linspace(0, 1, atmos1d.height.shape[0]), atmos1d.height)
temperature = np.zeros((Nz, Nx))
temperature[...] = weno4(z, oldZ, atmos1d.temperature)[:, None]
deltaT = 500
temperature += (deltaT * np.sin(6 * np.pi*x / Lx))[None, :]
vx = np.zeros((Nz, Nx))
vz = np.zeros((Nz, Nx))
vturb = np.zeros((Nz, Nx))
vturb[...] = weno4(z, oldZ, atmos1d.vturb)[:, None]
ne = np.zeros((Nz, Nx))
ne[...] = weno4(z, oldZ, atmos1d.ne)[:, None]
nHTot = np.zeros((Nz, Nx))
nHTot[...] = weno4(z, oldZ, atmos1d.nHTot)[:, None]
atmos = lw.Atmosphere.make_2d(height=z, x=x, temperature=temperature, vx=vx, vz=vz, vturb=vturb, ne=ne, nHTot=nHTot)

def atmos_hierarchy(atmos, Nlevels):
    result = [atmos]
    for i in range(Nlevels - 1):
        result.append(upsample_atmos_2d(result[-1]))

    return result

atmoses = atmos_hierarchy(atmos, 3)
for atmos in atmoses:
    atmos.quadrature(7)
baseAtmos = deepcopy(atmoses[-1])
baseAtmos.quadrature(7)

aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), Mg_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()

# molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
# mols = lw.MolecularTable(molPaths)

baseEqPops = aSet.compute_eq_pops(baseAtmos)
baseCtx = lw.Context(baseAtmos, spect, baseEqPops, Nthreads=8)
eqPopses = [aSet.compute_eq_pops(a) for a in atmoses]
ctxs = [lw.Context(a, spect, e, Nthreads=8) for a, e in zip(atmoses, eqPopses)]

eta1 = 1e-1
eta2 = 5e-4
nu1 = 2
nu2 = 4


Rc = 1.0
count = 0
global nIter
nIter = [0 for f in ctxs]
start = time.time()
for i in range(nu2):
    ctxs[-1].formal_sol_gamma_matrices()
while Rc > eta2:
    rhs = [np.zeros_like(eqPopses[-1][a.element]) for a in ctxs[-1].activeAtoms]
    for i, a in enumerate(ctxs[-1].activeAtoms):
        rhs[i][-1, :] = eqPopses[-1].atomicPops[a.element].nTotal
    Rc = v_cycle(ctxs, eqPopses, len(ctxs)-1, rhs)
    count += 1
    print('Rc after cycle %d, %e' % (count, Rc))
end = time.time()

baseError = 1.0
startBase = time.time()
for i in range(nu2):
    baseCtx.formal_sol_gamma_matrices()
while baseError > eta2:
    baseCtx.formal_sol_gamma_matrices()
    baseError = baseCtx.stat_equil()

endBase = time.time()

print('MG: %e' % (end - start))
print('Basic: %e' % (endBase - startBase))

# for i in range(2):
#     ctx.formal_sol_gamma_matrices()
# dPops = [1.0]
# for i in range(2000):
#     ctx.formal_sol_gamma_matrices(lambdaIterate=False)
#     dPops.append(ctx.stat_equil())
#     if dPops[-1] < 5e-4:
#         print(i)
#         break

# # dJs = []
# # for i in range(1000):
# #     dJs.append(ctx.formal_sol_gamma_matrices())
# #     if dJs[-1] < 1e-9:
# #         print(i)
# #         break