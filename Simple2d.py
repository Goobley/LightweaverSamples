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
# eqPops = aSet.iterate_lte_ne_eq_pops(atmos, mols)
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, formalSolver='piecewise_parabolic_2d')
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, interpFn='interp_linear_2d')
start = time.time()
ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, conserveCharge=False)
ctx.depthData.fill = True

for i in range(2):
    ctx.formal_sol_gamma_matrices()
dPops = [1.0]
n = eqPops['Ca']
for i in range(2000):
    ctx.formal_sol_gamma_matrices(lambdaIterate=False)
    dPops.append(ctx.stat_equil())
    ctx.prd_redistribute(maxIter=10, tol=dPops[-1])
    if np.any(n < 0):
        print(i)
        print('neg')
        break
    if dPops[-1] < 5e-4:
        print(i)
        break

end = time.time()
print('Time: %e s' % (end - start))

wave = np.linspace(853.9444, 854.9444, 1001)
eqPops.update_lte_atoms_Hmin_pops(atmos)
Iwave = ctx.compute_rays(wave, [1.0], stokes=False)