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
from lightweaver.utils import InitialSolution

def prep_atmos(data, xIdx, yIdx):
    height = data[xIdx, yIdx, :, 0].astype('<f8') / 1e2
    temp = data[xIdx, yIdx, :, 1].astype('<f8')
    vlos = data[xIdx, yIdx, :, 3].astype('<f8') / 1e2
    pgasTop = data[xIdx, yIdx, 0, 2].astype('<f8') / (Const.CM_TO_M**2 / Const.G_TO_KG)

    return {'height': height, 'temp': temp, 'vlos': vlos, 'pgasTop': pgasTop}

def iterate_ctx(ctx, prd=True, Nscatter=3, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if prd:
            dRho = ctx.prd_redistribute(maxIter=5)

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

wave = np.linspace(853.9444, 854.9444, 1001)

data = fits.getdata('better_eb_310400.fits')
atmosData = prep_atmos(data, 20,20)

atmos = Atmosphere(ScaleType.Geometric, depthScale=atmosData['height'], temperature=atmosData['temp'], vlos=atmosData['vlos'], vturb=4000*np.ones_like(atmosData['height']))

aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H', 'Ca')

spect = aSet.compute_wavelength_grid()

atmos.convert_scales(Ptop=atmosData['pgasTop'])
atmos.quadrature(5)

mols = MolecularTable()
eqPops = aSet.iterate_lte_ne_eq_pops(mols, atmos)
ctx = LwContext(atmos, spect, eqPops, conserveCharge=True, initSol=InitialSolution.Lte)
iterate_ctx(ctx, prd=False)
eqPops.update_lte_atoms_Hmin_pops(atmos)
Iwave = ctx.compute_rays(wave, [1.0])

plt.ion()
plt.plot(wave, Iwave)
plt.show()