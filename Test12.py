from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
import lightweaver as lw
# from PyProto import InitialSolution
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
from lightweaver.utils import NgOptions, get_default_molecule_path
from astropy.io import fits

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

wave = np.linspace(853.9444, 854.9444, 1001)
# wave = np.linspace(392, 398, 10001)
# wave = np.linspace(655.9691622298104, 656.9691622298104, 1001)
def synth_8542(atmos, conserve, useNe, stokes=False):
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
    mols = MolecularTable(molPaths)

    if useNe:
        eqPops = aSet.compute_eq_pops(atmos, mols)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos, mols)
    ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=True, conserveCharge=conserve, Nthreads=8)
    ctx.depthData.fill = True
    start = time.time()
    iterate_ctx(ctx, atmos, eqPops, prd=False, updateLte=False)
    end = time.time()
    print('%.2f s' % (end - start))
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    IwaveStokes = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=stokes)
    return ctx, Iwave, IwaveStokes

def add_B(atmos):
    atmos.B = np.ones(atmos.Nspace) * 1.0
    atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25
    atmos.chiB = np.zeros(atmos.Nspace)


atmos = Falc82()
atmos.convert_scales()
atmos.quadrature(5)
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
# aSet = RadiativeSet([H_6_atom(), CaII_atom()])
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()

neInit = np.copy(atmos.ne)
eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
# eqPops = aSet.compute_eq_pops(atmos)
neIterStart = np.copy(atmos.ne)

ctx = lw.Context(atmos, spect, eqPops, initSol=lw.InitialSolution.Lte, Nthreads=8)
for i in range(300):
    dJ = ctx.formal_sol_gamma_matrices()
    if i < 5:
        continue
    delta = ctx.stat_equil()
    dNr = ctx.nr_post_update(fdCollisionRates=True)
    # if prd:
    #     dRho = ctx.prd_redistribute(maxIter=5)

    # if updateLte:
    # eqPops.update_lte_atoms_Hmin_pops(atmos)
    # ctx.update_deps()
    for p in eqPops.atomicPops:
        p.nStar[:] = lw.lte_pops(p.model, atmos.temperature, atmos.ne, p.nTotal)

    if dJ < 3e-3 and delta < 1e-3:
        print(i)
        print('----------')
        break
ctx.update_deps()
Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)

atmosRef = Falc82()
atmosRef.convert_scales()
atmosRef.quadrature(5)
eqPopsRef = aSet.compute_eq_pops(atmosRef)
ctxRef = lw.Context(atmosRef, spect, eqPopsRef, initSol=lw.InitialSolution.Lte, Nthreads=8)
iterate_ctx(ctxRef, atmosRef, eqPopsRef, prd=False)
IwaveRef = ctxRef.compute_rays(wave, [atmos.muz[-1]], stokes=False)

atmosCons = Falc82()
add_B(atmosCons)
ctxCons, IwaveCons, _ = synth_8542(atmosCons, conserve=True, useNe=False, stokes=False)
# atmosLte = Falc82()
# add_B(atmosLte)
# ctx, IwaveLte, _ = synth_8542(atmosLte, conserve=False, useNe=False, stokes=False)
# atmosFal = Falc82()
# add_B(atmosFal)
# ctx, Iwave, IwaveStokes = synth_8542(atmosFal, conserve=False, useNe=True, stokes=True)