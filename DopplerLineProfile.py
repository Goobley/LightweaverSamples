from lightweaver.atomic_model import AtomicModel, AtomicLevel, AtomicLine, HydrogenicContinuum, VoigtLine, LinearCoreExpWings, LineType, LineProfileState, LineProfileResult
from lightweaver.broadening import VdwUnsold, QuadraticStarkBroadening, HydrogenLinearStarkBroadening, LineBroadening, LineBroadener, RadiativeBroadening
from lightweaver.collisional_rates import CE, CI
from fractions import Fraction
from dataclasses import dataclass
import lightweaver as lw
import numpy as np
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import matplotlib.pyplot as plt

@dataclass(eq=False, repr=False)
class DopplerLine(AtomicLine):
    def compute_phi(self, state: LineProfileState) -> LineProfileResult:
        vBroad = self.atom.vBroad(state.atmos) if state.vBroad is None else state.vBroad
        aDamp = np.zeros_like(vBroad)
        Qelast = np.zeros_like(vBroad)
        atmos = state.atmos
        vlosMu = state.vlosMu

        xBase = (state.wavelength - self.lambda0) * lw.CLight / self.lambda0
        phi = np.zeros((state.wavelength.shape[0], atmos.Nrays, 2, atmos.Nspace))

        for mu in range(vlosMu.shape[0]):
            for toObs, sign in enumerate([-1.0, 1.0]):
                for k in range(atmos.Nspace):
                    xk = (xBase + sign * vlosMu[mu, k]) / vBroad[k]
                    phi[:, mu, toObs, k] = np.exp(-xk**2) / (np.sqrt(np.pi) * vBroad[k])

        return LineProfileResult(phi=phi, aDamp=aDamp, Qelast=Qelast)

@dataclass
class NoOpBroadener(LineBroadener):
    def __eq__(self, other):
        if type(self) is type(other):
            return True
        return False

    def __repr__(self):
        return 'NoOpBroadener()'

def no_broadening():
    return LineBroadening(natural=[NoOpBroadener()], elastic=[NoOpBroadener()])

H_6_doppler = lambda: \
AtomicModel(element=lw.PeriodicTable['H'],
    levels=[
        AtomicLevel(E=     0.000, g=2, label="H I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
        AtomicLevel(E= 82258.211, g=8, label="H I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
        AtomicLevel(E= 97491.219, g=18, label="H I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
        AtomicLevel(E=102822.766, g=32, label="H I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
        AtomicLevel(E=105290.508, g=50, label="H I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
        AtomicLevel(E=109677.617, g=1, label="H II continuum", stage=1, J=None, L=None, S=None),
    ],
    lines=[
        DopplerLine(j=1, i=0, f=4.162e-01, type=LineType.PRD, quadrature=LinearCoreExpWings(qCore=15, qWing=600, Nlambda=100), broadening=no_broadening()),
        DopplerLine(j=2, i=0, f=7.910e-02, type=LineType.PRD, quadrature=LinearCoreExpWings(qCore=10, qWing=250, Nlambda=50), broadening=no_broadening()),
        DopplerLine(j=3, i=0, f=2.899e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=20), broadening=no_broadening()),
        DopplerLine(j=4, i=0, f=1.394e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=20), broadening=no_broadening()),
        DopplerLine(j=2, i=1, f=6.407e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=250, Nlambda=70), broadening=no_broadening()),
        DopplerLine(j=3, i=1, f=1.193e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=250, Nlambda=40), broadening=no_broadening()),
        DopplerLine(j=4, i=1, f=4.467e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=250, Nlambda=40), broadening=no_broadening()),
        DopplerLine(j=3, i=2, f=8.420e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=30, Nlambda=20), broadening=no_broadening()),
        DopplerLine(j=4, i=2, f=1.506e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=30, Nlambda=20), broadening=no_broadening()),
        DopplerLine(j=4, i=3, f=1.036e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=30, Nlambda=20), broadening=no_broadening()),
    ],
    continua=[
        HydrogenicContinuum(j=5, i=0, NlambdaGen=20, alpha0=6.152e-22, minWavelength=22.794),
        HydrogenicContinuum(j=5, i=1, NlambdaGen=20, alpha0=1.379e-21, minWavelength=91.176),
        HydrogenicContinuum(j=5, i=2, NlambdaGen=20, alpha0=2.149e-21, minWavelength=205.147),
        HydrogenicContinuum(j=5, i=3, NlambdaGen=20, alpha0=2.923e-21, minWavelength=364.705),
        HydrogenicContinuum(j=5, i=4, NlambdaGen=20, alpha0=3.699e-21, minWavelength=569.852),
    ],
    collisions=[
        CE(j=1, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[9.75e-16, 6.098e-16, 4.535e-16, 3.365e-16, 2.008e-16, 1.56e-16]),
        CE(j=2, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[1.437e-16, 9.069e-17, 6.798e-17, 5.097e-17, 3.118e-17, 2.461e-17]),
        CE(j=3, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[4.744e-17, 3.001e-17, 2.255e-17, 1.696e-17, 1.044e-17, 8.281e-18]),
        CE(j=4, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[2.154e-17, 1.364e-17, 1.026e-17, 7.723e-18, 4.772e-18, 3.791e-18]),
        CE(j=2, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[1.127e-14, 8.077e-15, 6.716e-15, 5.691e-15, 4.419e-15, 3.89e-15]),
        CE(j=3, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[1.36e-15, 1.011e-15, 8.617e-16, 7.482e-16, 6.068e-16, 5.484e-16]),
        CE(j=4, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[4.04e-16, 3.041e-16, 2.612e-16, 2.287e-16, 1.887e-16, 1.726e-16]),
        CE(j=3, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[3.114e-14, 2.629e-14, 2.434e-14, 2.29e-14, 2.068e-14, 1.917e-14]),
        CE(j=4, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[3.119e-15, 2.7e-15, 2.527e-15, 2.4e-15, 2.229e-15, 2.13e-15]),
        CE(j=4, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[7.728e-14, 7.317e-14, 7.199e-14, 7.109e-14, 6.752e-14, 6.31e-14]),
        CI(j=5, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[2.635e-17, 2.864e-17, 3.076e-17, 3.365e-17, 4.138e-17, 4.703e-17]),
        CI(j=5, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[5.34e-16, 6.596e-16, 7.546e-16, 8.583e-16, 1.025e-15, 1.069e-15]),
        CI(j=5, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[2.215e-15, 2.792e-15, 3.169e-15, 3.518e-15, 3.884e-15, 3.828e-15]),
        CI(j=5, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[6.182e-15, 7.576e-15, 8.37e-15, 8.992e-15, 9.252e-15, 8.752e-15]),
        CI(j=5, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0], rates=[1.342e-14, 1.588e-14, 1.71e-14, 1.786e-14, 1.743e-14, 1.601e-14]),
])


def iterate_ctx(ctx, Nscatter=3, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

wave = np.linspace(656, 657, 1001)
def synth_halpha(atmos, dopplerLines=False):
    atmos.convert_scales()
    atmos.quadrature(5)

    Hatom = H_6_doppler if dopplerLines else H_6_atom
    aSet = lw.RadiativeSet([Hatom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=8)
    iterate_ctx(ctx)
    Iwave = ctx.compute_rays(wave, [1.0], stokes=False)
    return ctx, Iwave

atmosRef = Falc82()
ctxRef, IwaveRef  = synth_halpha(atmosRef, dopplerLines=False)
atmosDopp = Falc82()
ctxDopp, IwaveDopp = synth_halpha(atmosDopp, dopplerLines=True)

plt.ion()
plt.plot(wave, IwaveRef, label='Voigt')
plt.plot(wave, IwaveDopp, label='Doppler')
plt.legend()