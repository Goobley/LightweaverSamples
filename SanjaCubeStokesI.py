import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import lightweaver.constants as Const
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.atmosphere import Atmosphere, ScaleType
from lightweaver.atomic_set import RadiativeSet
from lightweaver.atomic_table import get_global_atomic_table
from lightweaver.molecule import MolecularTable
from lightweaver.LwCompiled import LwContext
from lightweaver.utils import InitialSolution, ExplodingMatrixError, ConvergenceError
import lightweaver as lw
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
from tqdm import tqdm
from contextlib import redirect_stdout
import os
import socket
import pickle
from notify_run import Notify

def prep_atmos(data, extData, xIdx, yIdx):
    height = data[xIdx, yIdx, :, 0].astype('<f8') / 1e2
    temp = data[xIdx, yIdx, :, 1].astype('<f8')
    vlos = data[xIdx, yIdx, :, 3].astype('<f8') / 1e2
    # pgasTop = data[xIdx, yIdx, 0, 2].astype('<f8') / (Const.CM_TO_M**2 / Const.G_TO_KG)
    pgas = data[xIdx, yIdx, :, 2].astype('<f8') / (Const.CM_TO_M**2 / Const.G_TO_KG)
    density = extData[xIdx, yIdx, :, 1].astype('<f8') * 1e3
    ne = extData[xIdx, yIdx, :, 2].astype('<f8') * 1e6

    return {'x': xIdx, 'y': yIdx, 'height': height, 'temp': temp, 'vlos': vlos, 'pgas': pgas, 'density': density, 'ne': ne}

def iterate_ctx(ctx, eqPops, prd=True, Nscatter=3, NmaxIter=1000, nr=False):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if nr:
            ctx.nr_post_update()
            for p in eqPops.atomicPops:
                p.nStar[:] = lw.lte_pops(p.model, atmos.temperature, atmos.ne, p.nTotal)
        if prd:
            dRho = ctx.prd_redistribute(maxIter=5)

        if np.any(np.isnan(ctx.spect.J)):
            raise ExplodingMatrixError('NaNs abound!')

        if ctx.crswDone and dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return i

    raise ConvergenceError('Context not converged after %d iterations' % NmaxIter)

wave = np.linspace(853.9444, 854.9444, 1001)

data = fits.getdata('better_eb_310400.fits')
extData = fits.getdata('better_eb_310400_ext.fits')
# atmosData = prep_atmos(data, 10,10)

def crsw_factory(initVal=1e3):
    val = initVal
    def callback():
        nonlocal val
        val = max(1.0, val * 0.1**(1/val))
        return val
    return callback

def cmo_synth(atmosData, crsw=None, NmaxIter=1000):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            if crsw is not None:
                crsw = crsw()
            atmos = Atmosphere(ScaleType.Geometric, depthScale=atmosData['height'], temperature=atmosData['temp'], vlos=atmosData['vlos'], vturb=4000*np.ones_like(atmosData['height']))

            aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('H', 'Ca')

            spect = aSet.compute_wavelength_grid()

            atmos.convert_scales(Pgas=atmosData['pgas'])
            atmos.quadrature(5)

            eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
            ctx = lw.Context(atmos, spect, eqPops, conserveCharge=False, initSol=InitialSolution.Lte, crswCallback=crsw)
            converged = True
            exploding = False
            try:
                nIter = iterate_ctx(ctx, eqPops, prd=False, NmaxIter=NmaxIter, nr=True)
            except ConvergenceError:
                converged = False
                nIter = NmaxIter
            except ExplodingMatrixError:
                converged = False
                exploding = True
                nIter = NmaxIter

            if converged:
                eqPops.update_lte_atoms_Hmin_pops(atmos)
                Iwave = ctx.compute_rays(wave, [1.0])
                return {'Iwave': Iwave, 'eqPops': eqPops, 'converged': converged, 'nIter': nIter, 'atmosData': atmosData, 'exploding': exploding}
            else:
                return {'converged': converged, 'nIter': nIter, 'atmosData': atmosData, 'exploding': exploding}

def cmo_synth_lte(atmosData):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            atmos = Atmosphere(ScaleType.Geometric, depthScale=atmosData['height'], temperature=atmosData['temp'], vlos=atmosData['vlos'], vturb=4000*np.ones_like(atmosData['height']))

            aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('Ca')

            spect = aSet.compute_wavelength_grid()

            atmos.convert_scales(Pgas=atmosData['pgas'])
            atmos.quadrature(5)

            eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
            ctx = LwContext(atmos, spect, eqPops, conserveCharge=False)
            iterate_ctx(ctx, eqPops, prd=False)
            Iwave = ctx.compute_rays(wave, [1.0])
            return Iwave

def cmo_synth_given(atmosData):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            at = get_global_atomic_table()
            atmos = Atmosphere(ScaleType.Geometric, depthScale=atmosData['height'], temperature=atmosData['temp'], vlos=atmosData['vlos'], vturb=4000*np.ones_like(atmosData['height']),  ne=atmosData['ne'], nHTot=(atmosData['density'] / (at.weightPerH*Const.Amu)))

            aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
            aSet.set_active('Ca')

            spect = aSet.compute_wavelength_grid()

            atmos.convert_scales()
            atmos.quadrature(5)

            eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
            ctx = LwContext(atmos, spect, eqPops, conserveCharge=False)
            iterate_ctx(ctx, eqPops, prd=False)
            Iwave = ctx.compute_rays(wave, [1.0])
            return Iwave

atmosData = []
for x in range(0, data.shape[0]):
    for y in range(0, data.shape[1]):
        atmosData.append(prep_atmos(data, extData, x, y))

del data, extData

redoJobs = []
redoFutures = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(cmo_synth, d) for d in atmosData[:3]]
    tq = tqdm(as_completed(futures), total=len(futures))
    for f in tq:
        try:
            pass
            # res = f.result()
            # if not res['converged']:
            #     redoJobs.append({'atmosData': res['atmosData'], 'NmaxIter': res['nIter']*2})
            # if res['exploding']:
            #     redoJobs.append({'atmosData': res['atmosData'], 'crsw': crsw_factory})
        except:
            pass

    if len(redoJobs) > 0:
        redoFutures = [executor.submit(cmo_synth, **params) for params in redoJobs]
        tq = tqdm(as_completed(redoFutures), total=len(redoFutures))
        for f in tq:
            pass


spectra = []
for f in futures + redoFutures:
    try:
        spectra.append(f.result())
    except:
        spectra.append(None)

name = 'HerculesNrCube.pickle'
with open(name, 'wb') as f:
    pickle.dump(spectra, f)

notify = Notify()
notify.read_config()
notify.send('%s <%s> done!' % (__file__, socket.gethostname()))



# atmosHse = Atmosphere(ScaleType.Geometric, depthScale=atmosData['height'], temperature=atmosData['temp'], vlos=atmosData['vlos'], vturb=4000*np.ones_like(atmosData['height']))


# atmosHse.convert_scales(Ptop=atmosData['pgas'][0])
# atmosHse.quadrature(5)

# aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
# aSet.set_active('Ca')
# spectHse = aSet.compute_wavelength_grid()
# eqPopsHse = aSet.iterate_lte_ne_eq_pops(mols, atmosHse)
# ctxHse = LwContext(atmosHse, spectHse, eqPopsHse, conserveCharge=False, initSol=InitialSolution.Lte)
# iterate_ctx(ctxHse, prd=False)
# eqPopsHse.update_lte_atoms_Hmin_pops(atmosHse)
# IwaveHse = ctxHse.compute_rays(wave, [1.0])

# plt.ion()
# plt.plot(wave, Iwave)
# plt.plot(wave, IwaveHse)
# plt.show()