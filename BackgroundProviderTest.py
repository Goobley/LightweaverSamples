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

from lightweaver.LwCompiled import BackgroundProvider
from lightweaver.witt import witt

class WittmanBackground(BackgroundProvider):
    """
    Uses the background from the Wittmann EOS, ignores scattering (i.e. sets to 0)
    """
    def __init__(self, eqPops, radSet, wavelength):
        self.eqPops = eqPops
        self.radSet = radSet
        self.wavelength = wavelength

    def compute_background(self, atmos, chi, eta, sca):
        abundance = self.eqPops.abundance
        wittAbundances = np.array([abundance[e] for e in lw.PeriodicTable.elements])
        eos = witt(abund_init=wittAbundances)

        rhoCgs = lw.Amu * abundance.massPerH * atmos.nHtot * lw.CM_TO_M**3 / lw.G_TO_KG
        for k in range(atmos.temperature.shape[0]):
            pgas = eos.pg_from_rho(atmos.temperature[k], rhoCgs[k])
            pe   = eos.pe_from_rho(atmos.temperature[k], rhoCgs[k])
            chiC = eos.contOpacity(atmos.temperature[k], pgas, pe, self.wavelength * 10) / lw.CM_TO_M
            chi[:, k] = chiC
            Bnu = lw.planck(atmos.temperature[k], self.wavelength)
            eta[:, k] = Bnu * chiC

        # sca[...] = 1e-2 * chi
        sca[...] = 0.0

def iterate_ctx(ctx, Nscatter=5, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()

        if dJ < 3e-3 and delta < 1e-3 and i > 2 * Nscatter:
            print(i)
            print('----------')
            return

wave = np.linspace(853.9444, 854.9444, 1001)
def synth_8542(atmos, backgroundProvider=None):
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('Ca')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=8, backgroundProvider=backgroundProvider)
    iterate_ctx(ctx)
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    return ctx, Iwave

atmosWitt = Falc82()
ctxWitt, IwaveWitt = synth_8542(atmosWitt, backgroundProvider=WittmanBackground)
atmosRef = Falc82()
ctxRef, IwaveRef = synth_8542(atmosRef, backgroundProvider=None)

plt.ion()
plt.plot(wave, IwaveRef, label='RH-Style')
plt.plot(wave, IwaveWitt, label='Wittmann')