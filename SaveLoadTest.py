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
from os import path

def iterate_ctx(ctx, prd=True, Nscatter=3, NmaxIter=200):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if prd:
            dRho = ctx.prd_redistribute(maxIter=6, tol=1e-6)

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

wave = np.linspace(279, 280, 5001)
def make_context():
    atmos = Falc82()
    set_vel(atmos)
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca', 'Mg')
    spect = aSet.compute_wavelength_grid()

    molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
    mols = lw.MolecularTable(molPaths)

    eqPops = aSet.compute_eq_pops(atmos, mols)
    ctx = lw.Context(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=True, Nthreads=8)
    return ctx

def compute_mgk(ctx, maxIter=200):
    iterate_ctx(ctx, prd=True, NmaxIter=maxIter)
    Iwave = ctx.compute_rays(wave, 1.0, stokes=False)
    return Iwave

def set_vel(atmos):
    atmos.vlos[:] = np.sin(np.linspace(0, 12*np.pi, atmos.Nspace)) * 20e3

ctxFilename = 'SaveCtxTest.pickle'
refSpectFilename = 'SaveCtxIwaveRef.npy'
startingIterations = 30
if path.isfile(ctxFilename):
    print('Loading Context')
    with open(ctxFilename, 'rb') as pkl:
        ctx = pickle.load(pkl)

    Iwave = compute_mgk(ctx)
    IwaveRef = np.load(refSpectFilename)
    plt.ion()
    plt.plot(wave, Iwave, label='Loaded Ctx')
    plt.plot(wave, IwaveRef, '--', label='Reference')

else:
    print('Generating Reference Solution')
    ctxRef = make_context()
    IwaveRef = compute_mgk(ctxRef)
    np.save(refSpectFilename, IwaveRef)

    ctxSave = make_context()
    iterate_ctx(ctxSave, prd=True, NmaxIter=startingIterations)
    with open(ctxFilename, 'wb') as pkl:
        pickle.dump(ctxSave, pkl)
    print('Saved Context after %d iterations' % startingIterations)