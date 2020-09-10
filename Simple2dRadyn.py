import os, sys
sys.path.append(os.environ['HOME'] + '/MsLightweaver/')
from MsLightweaverAtoms import H_6, CaII
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, Mg_atom, N_atom, Na_atom, S_atom
from weno4 import weno4
from radynpy.cdf import LazyRadynData
import numpy as np
import time

radyn = LazyRadynData(os.environ['HOME'] + '/MsLightweaverMeeting27022020/radyn_out.cdf')

Nz = 300
zGrid = np.interp(np.linspace(0,1,Nz), np.linspace(0,1,radyn.z1.shape[1]), radyn.z1[0])
Nx = 101
XWidth = 1e6
xGrid = np.linspace(0, XWidth, Nx)

temperature = np.zeros((Nz, Nx))
temperature[...] = weno4(zGrid, radyn.z1[0], radyn.tg1[0])[:, None]
vx = np.zeros((Nz, Nx))
vz = np.zeros((Nz, Nx))
vturb = np.ones((Nz, Nx)) * 2e3
ne = np.zeros((Nz, Nx))
ne[...] = weno4(zGrid, radyn.z1[0], radyn.ne1[0] * 1e6)[:, None]
nHTot = np.zeros((Nz, Nx))
nHTot[...] = weno4(zGrid, radyn.z1[0], radyn.d1[0] / lw.DefaultAtomicAbundance.massPerH / lw.Amu * 1e3)[:, None]
atmos = lw.Atmosphere.make_2d(height=zGrid, x=xGrid, temperature=temperature, vx=vx,
                              vz=vz, vturb=vturb, ne=ne, nHTot=nHTot)
atmos.quadrature(7)

aSet = lw.RadiativeSet([H_6(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII(), Fe_atom(), He_atom(), Mg_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)
# eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
atmos.hPops = eqPops['H']
atmos.bHeat = np.zeros(Nx*Nz)
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, formalSolver='piecewise_parabolic_2d')
# ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, interpFn='interp_linear_2d')
start = time.time()
ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, conserveCharge=False)

for i in range(3):
    ctx.formal_sol_gamma_matrices()
dPops = [1.0]
for i in range(2000):
    ctx.formal_sol_gamma_matrices(lambdaIterate=False)
    dPops.append(ctx.stat_equil())
    if dPops[-1] < 1e-3:
        print(i)
        break