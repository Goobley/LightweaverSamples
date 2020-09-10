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
from lightweaver.LwCompiled import FormalSolvers
from weno4 import weno4



# FormalSolvers.load_fs_from_path('../Lightweaver/Source/Parabolic2d.so')

abund = lw.AtomicAbundance({lw.PeriodicTable[k]: v for k, v in {
                            'C' : 8.39,
                            'O' : 8.66,
                            'Si': 7.55,
                            'Al': 6.47,
                            'Ca': 6.36,
                            'Fe': 7.44,
                            'He': 10.99,
                            'Mg': 7.58,
                            'N' : 8.00,
                            'Na': 6.33,
                            'S' : 7.21
                            }.items()})
atmos1d = Falc82()
x = (np.arange(5) * 100_000).astype(np.float64)
Nx = x.shape[0]
oldZ = np.copy(atmos1d.height)
z = weno4(np.linspace(0, 1, 83), np.linspace(0, 1, atmos1d.height.shape[0]), atmos1d.height)
Nz = z.shape[0]
temperature = np.zeros((Nz, Nx))
temperature[...] = weno4(z, oldZ, atmos1d.temperature)[:, None]
vx = np.zeros((Nz, Nx))
vz = np.zeros((Nz, Nx))
vturb = np.zeros((Nz, Nx))
vturb[...] = weno4(z, oldZ, atmos1d.vturb)[:, None]
ne = np.zeros((Nz, Nx))
ne[...] = weno4(z, oldZ, atmos1d.ne)[:, None]
nHTot = np.zeros((Nz, Nx))
nHTot[...] = weno4(z, oldZ, atmos1d.nHTot)[:, None]
atmos = lw.Atmosphere.make_2d(height=z, x=x, temperature=temperature, vx=vx, vz=vz, vturb=vturb, ne=ne, nHTot=nHTot)
# atmos.angle_set_a4()
atmos.quadrature(7)


aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), Mg_atom(), N_atom(), Na_atom(), S_atom()], abundance=abund)
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()

molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
mols = lw.MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(atmos, mols)
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, formalSolver='piecewise_parabolic_2d')
ctx = lw.Context(atmos, spect, eqPops, Nthreads=8)
# ctx.depthData.fill = True

for i in range(2):
    ctx.formal_sol_gamma_matrices()
dPops = [1.0]
Js = []
n = eqPops['Ca']
for i in range(2000):
    ctx.formal_sol_gamma_matrices(lambdaIterate=False)
    Js.append(np.copy(ctx.spect.J))
    dPops.append(ctx.stat_equil())
    if np.any(n < 0):
        print(i)
        print('neg')
        break
    if dPops[-1] < 1e-3:
        print(i)
        break

prevT = np.copy(atmos.temperature)
for i in range(11, 31):
    di = (i - 20.0) / 3.0
    atmos.dimensioned_view().temperature[i, Nx // 2] *= 1.0 + 2.0 * np.exp(-di**2)

dt = 0.1
NtStep = int(15.0 // dt)
NsubStep = 150

caPops = [np.copy(eqPops['Ca'])]
J = [np.copy(ctx.spect.J)]
I = [np.copy(ctx.spect.I)]
for it in range(NtStep):
    # eqPops.update_lte_atoms_Hmin_pops(atmos, conserveCharge=True)
    # ctx.background.update_background()
    # ctx.compute_profiles()
    ctx.update_deps()

    prevState = None
    for sub in range(NsubStep):
        dJ = ctx.formal_sol_gamma_matrices()
        delta, prevState = ctx.time_dep_update(dt, prevState)

        if delta < 1e-3 and dJ < 3e-3:
            break
    else:
        raise ValueError('No converge')

    ctx.time_dep_conserve_charge(prevState)
    caPops.append(np.copy(eqPops['ca']))
    J.append(np.copy(ctx.spect.J))
    I.append(np.copy(ctx.spect.I))
    print('Iteration %d (%f s) done after %d sub iterations' % (it, (it+1)*dt, sub))

