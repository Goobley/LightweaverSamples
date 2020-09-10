from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
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


# atmos1d = Falc82()
x = (np.linspace(0, 1, 6)) * 200000
Nx = x.shape[0]
# z = atmos1d.height
z = np.linspace(1, 0, 6) * 100000
Nz = z.shape[0]
temperature = np.zeros((Nx, Nz))
temperature[...] = 5000
vx = np.zeros((Nx, Nz))
vz = np.zeros((Nx, Nz))
vturb = np.zeros((Nx, Nz))
vturb[...] = 2000
ne = np.zeros((Nx, Nz))
ne[...] = 1e18
nHTot = np.zeros((Nx, Nz))
nHTot[...] = 1e18
atmos = lw.Atmosphere.make_2d(height=z, x=x, temperature=temperature, vx=vx, vz=vz, vturb=vturb, ne=ne, nHTot=nHTot)
atmos.angle_set_a4()

aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()

molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
mols = lw.MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(atmos, mols)
ctx = lw.Context(atmos, spect, eqPops, Nthreads=1)
ctx.depthData.fill = True

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
    if dPops[-1] < 5e-4:
        print(i)
        break

# dJs = []
# for i in range(1000):
#     dJs.append(ctx.formal_sol_gamma_matrices())
#     if dJs[-1] < 1e-9:
#         print(i)
#         break