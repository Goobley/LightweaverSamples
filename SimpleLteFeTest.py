from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom, Fe23_atom
import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
from lightweaver.utils import NgOptions, get_default_molecule_path


wave = np.linspace(630.2, 630.5, 1001)
def synth_fe(atmos, stokes=False):
    atmos.convert_scales() # NOTE(cmo): The slow bit!
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe23_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_detailed_static('Fe')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=8)
    for i in range(100):
        dJ = ctx.formal_sol_gamma_matrices()
        if dJ < 1e-3:
            print('%d iterations' % i)
            break

    Iwave = ctx.compute_rays(wave, [1.0], stokes=False)
    IwaveStokes = ctx.compute_rays(wave, [1.0], stokes=stokes)
    return ctx, Iwave, IwaveStokes

def add_B(atmos):
    atmos.B = np.ones(atmos.Nspace) * 0.5 # NOTE(cmo): 0.5 T !!
    atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25 # NOTE(cmo): Inclination of pi/4 (from vertical)
    atmos.chiB = np.zeros(atmos.Nspace) # NOTE(cmo): Azimuth from e_1 / x

atmos = Falc82()
add_B(atmos)
ctxRef, Iwave, IwaveStokes = synth_fe(atmos, stokes=True)

plt.ion()
fig, ax = plt.subplots(2, 1)
ax[0].plot(wave, Iwave, label='FALC Scalar')
ax[0].plot(wave, IwaveStokes[0], label='FALC Polarised I')
ax[1].plot(wave, IwaveStokes[1] / IwaveStokes[0, 0], label='Q / I_c')
ax[1].plot(wave, IwaveStokes[2] / IwaveStokes[0, 0], label='U / I_c')
ax[1].plot(wave, IwaveStokes[3] / IwaveStokes[0, 0], label='V / I_c')
ax[0].legend()
ax[1].legend()
