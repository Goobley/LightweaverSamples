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

FormalSolvers.load_fs_from_path('../Lightweaver/Source/Parabolic2d.so')
atmos1d = Falc82()
x = (np.arange(5) * 10_000).astype(np.float64)
Nx = x.shape[0]
z = atmos1d.height
Nz = z.shape[0]
temperature = np.zeros((Nz, Nx))
temperature[...] = atmos1d.temperature[:, None]
vx = np.zeros((Nz, Nx))
vz = np.zeros((Nz, Nx))
vturb = np.zeros((Nz, Nx))
vturb[...] = atmos1d.vturb[:, None]
ne = np.zeros((Nz, Nx))
ne[...] = atmos1d.ne[:, None]
nHTot = np.zeros((Nz, Nx))
nHTot[...] = atmos1d.nHTot[:, None]
atmos = lw.Atmosphere.make_2d(height=z, x=x, temperature=temperature, vx=vx, vz=vz, vturb=vturb, ne=ne, nHTot=nHTot)
atmos.angle_set_a4()


aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), Mg_atom(), N_atom(), Na_atom(), S_atom()], abundance=abund)
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()

molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
mols = lw.MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(atmos, mols)
ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, formalSolver='piecewise_parabolic_2d')
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8)
ctx.depthData.fill = True

# atmos1d.quadrature(5)
# eqPops1d = aSet.compute_eq_pops(atmos1d, mols)
# ctx1d = lw.Context(atmos1d, spect, eqPops1d, Nthreads=8)
# for i in range(2000):
#     ctx1d.formal_sol_gamma_matrices()
#     dPop = ctx1d.stat_equil()
#     if dPop < 5e-4:
#         print(i)
#         break
# Iwave1d = ctx1d.compute_rays(ctx.spect.wavelength, [atmos.muz[-1]])

start = time.time()
for i in range(6):
    ctx.formal_sol_gamma_matrices()
dPops = [1.0]
Js = []
n = eqPops['Ca']
for i in range(2000):
    ctx.formal_sol_gamma_matrices(lambdaIterate=False)
    # Js.append(np.copy(ctx.spect.J))
    dPops.append(ctx.stat_equil())
    # if np.any(n < 0):
    #     print(i)
    #     print('neg')
    #     break
    # if np.any(ctx.spect.I < 0):
    #     print(i)
    #     print('I neg')
    #     break
    if dPops[-1] < 5e-4:
        print(i)
        break
end = time.time()
print('%f s' % (end - start))



# dJs = []
# for i in range(1000):
#     dJs.append(ctx.formal_sol_gamma_matrices())
#     if dJs[-1] < 1e-9:
#         print(i)
#         break