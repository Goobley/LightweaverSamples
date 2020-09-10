from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, He_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
import time

def iterate_ctx(ctx, Nscatter=3, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        # NOTE(cmo): Do some initial iterations without touching the
        # populations to lambda iterate the background scattering terms
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()

        # NOTE(cmo): Check convergence
        if dJ < 3e-3 and delta < 1e-3:
            print('Iterations taken: %d' % (i+1))
            print('-'*60)
            return

wave = np.linspace(853.9444, 854.9444, 1001)
def synth_8542(atmos, conserve, useNe):
    # NOTE(cmo): Configure the Gauss-Legendre quadrature for 5 rays
    atmos.quadrature(5)

    # NOTE(cmo): Construct the RadiativeSet with the following atomic models
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    # NOTE(cmo): Set Hydrogen and Calcium to active
    aSet.set_active('H', 'Ca')
    # NOTE(cmo): Compute the SpectrumConfiguration for this RadiativeSet
    spect = aSet.compute_wavelength_grid()

    # NOTE(cmo): If we're using the electron density provided with FAL C, then
    # compute the associated LTE populations, otherwise find a solution for
    # self consistent LTE populations and electron density.
    if useNe:
        eqPops = aSet.compute_eq_pops(atmos)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

    # NOTE(cmo): Construct the Context, optionally setting chargeConservation and the number of threads to use.
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=1)

    # NOTE(cmo): Iterate the NLTE problem to convergence
    iterate_ctx(ctx)
    # NOTE(cmo): Compute a detailed solution to Ca II 8542 on the 1 nm wavelength grid above
    Iwave = None
    # Iwave = ctx.compute_rays(wave, [1.0], stokes=False).squeeze()
    return Iwave

# NOTE(cmo): Load an atmosphere. In this case we include a copy of FAL C, but
# Lightweaver also supports loading atmospheres in the MULTI format, and it is
# also simple to do so from the raw data components
Nruns = 1
start = []
mid = []
end = []
for i in range(Nruns):
    start.append(time.time())
    atmosRef = Falc82()
    mid.append(time.time())
    # NOTE(cmo): Ca II 8542 with the reference electron density in the FAL C atmosphere
    IwaveRef = synth_8542(atmosRef, conserve=False, useNe=True)
    end.append(time.time())

start = np.array(start)
mid = np.array(mid)
end = np.array(end)

# start.append(time.time())
# atmosCons = Falc82()
# # NOTE(cmo): Ca II 8542 with the electron density obtained from charge conservation
# mid.append(time.time())
# IwaveCons = synth_8542(atmosCons, conserve=True, useNe=False)
# end.append(time.time())

# start.append(time.time())
# atmosLte = Falc82()
# # NOTE(cmo): Ca II 8542 with LTE electron density
# mid.append(time.time())
# IwaveLte = synth_8542(atmosLte, conserve=False, useNe=False)
# end.append(time.time())

# from helita.sim.rh import Rhout

# rh = Rhout('../VanillaRh/rhf1d/run/')
# rh.read_ray('../VanillaRh/rhf1d/run/spectrum_1.00')

# # plt.ion()
# plt.plot(rh.wave, rh.int, label='RH FAL $n_e$')
# plt.plot(wave, IwaveRef, '--', label='Lightweaver FAL $n_e$')
# plt.plot(wave, IwaveCons, label='Lightweaver Charge Cons')
# plt.plot(wave, IwaveLte, label='Lightweaver LTE $n_e$')
# plt.xlabel(r'$\lambda$ [$nm$]')
# plt.ylabel(r'Intensity [$J\,m^{-2}\,s^{-1}\,Hz^{-1}\,sr^{-1}$]')
# plt.xlim(wave.min(), wave.max())
# plt.legend()
# plt.savefig('RhComparison.png', dpi=300)

for i in range(Nruns):
    print('%.4e, %.4e' % (end[i] - start[i], end[i] - mid[i]))
print('%.4e, %.4e' % (np.mean(end - start), np.mean(end - mid)))