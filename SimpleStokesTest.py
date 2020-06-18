from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
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
def synth_8542(atmos, conserve, useNe, stokes=False):
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
    mols = lw.MolecularTable(molPaths)

    if useNe:
        eqPops = aSet.compute_eq_pops(atmos, mols)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos, mols)
    ctx = lw.Context(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), conserveCharge=conserve, Nthreads=8)
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



atmosRef = Falc82()
add_B(atmosRef)
ctxRef, IwaveRef, stokesRef = synth_8542(atmosRef, conserve=False, useNe=True, stokes=True)
atmosCons = Falc82()
add_B(atmosCons)
ctxCons, IwaveCons, stokesCons = synth_8542(atmosCons, conserve=True, useNe=False, stokes=True)
atmosLte = Falc82()
add_B(atmosLte)
ctx, IwaveLte, stokesLte = synth_8542(atmosLte, conserve=False, useNe=False, stokes=True)

plt.ion()
plt.plot(wave, stokesRef[-1], label='V Reference FAL')
plt.plot(wave, stokesCons[-1], label='V Reference Cons')
plt.plot(wave, stokesLte[-1], label='V Reference LTE n_e')

plt.legend()