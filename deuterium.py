from lightweaver.atomic_table import PeriodicTable
from lightweaver.atomic_model import AtomicModel, AtomicLevel, VoigtLine, HydrogenicContinuum, LinearCoreExpWings, LineType
from lightweaver.collisional_rates import CE, CI
from lightweaver.broadening import LineBroadening, RadiativeBroadening, VdwUnsold, QuadraticStarkBroadening, HydrogenLinearStarkBroadening
from fractions import Fraction
## protium ionization potential 13.59844 eV, Lide, D.R. (Editor), Ionization potentials of atoms and atomic ions in Handbook of Chem. and Phys., 1992, 10-211.
## 13.59844 / (h * c) = 109678.815323 / cm

## deuterium ionization potential 13.603 eV, Kelly, R.L., Atomic and ionic spectrum lines of hydrogen through kryton, J. Phys. Chem. Ref. Data, 1987, 16.
## 13.603 / (h * c) = 109715.594204 / cm

## oscillator strengths, binding energies, and spontaneous emission coefficients taken from Wiese, W.~L., \& Fuhr, J.~R.\ 2009, Journal of Physical and Chemical Reference Data, 38, 565


D_6_atom = lambda: \
AtomicModel(element=PeriodicTable['D'],
            levels=[
                    AtomicLevel(E=0.000000, g=2.000000, label="D I 1S 2SE", stage=0, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
                    AtomicLevel(E=82281.545000, g=8.000000, label="D I 2P 2PO", stage=0, J=Fraction(7, 2), L=1, S=Fraction(1, 2)),
                    AtomicLevel(E=97518.810000, g=18.000000, label="D I 3D 2DE", stage=0, J=Fraction(17, 2), L=2, S=Fraction(1, 2)),
                    AtomicLevel(E=102851.857000, g=32.000000, label="D I 4F 2FO", stage=0, J=Fraction(31, 2), L=3, S=Fraction(1, 2)),
                    AtomicLevel(E=105320.293000, g=50.000000, label="D I 5G 2GE", stage=0, J=Fraction(49, 2), L=4, S=Fraction(1, 2)),
                    AtomicLevel(E=109715.594204, g=1.000000, label="D II continuum", stage=1, J=None, L=None, S=None),
                    ],
            lines=[
                   VoigtLine(j=1, i=0, f=4.163000e-01, type=LineType.PRD,
                             quadrature=LinearCoreExpWings(Nlambda=100, qCore=15.000000, qWing=600.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(4.699900e+08)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=2, i=0, f=7.912100e-02, type=LineType.PRD,
                             quadrature=LinearCoreExpWings(Nlambda=50, qCore=10.000000, qWing=250.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(9.987900e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=3, i=0, f=2.899800e-02, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=20, qCore=3.000000, qWing=100.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(3.019810e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=4, i=0, f=1.394200e-02, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=20, qCore=3.000000, qWing=100.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(1.155860e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=2, i=1, f=6.408900e-01, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=70, qCore=3.000000, qWing=250.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(9.987900e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=3, i=1, f=1.193500e-01, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=40, qCore=3.000000, qWing=250.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(3.019810e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=4, i=1, f=4.468100e-02, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=40, qCore=3.000000, qWing=250.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(1.155860e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=3, i=2, f=8.422200e-01, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=20, qCore=2.000000, qWing=30.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(3.019810e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=4, i=2, f=1.506100e-01, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=20, qCore=2.000000, qWing=30.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(1.155860e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   VoigtLine(j=4, i=3, f=1.037800e+00, type=LineType.CRD,
                             quadrature=LinearCoreExpWings(Nlambda=20, qCore=1.000000, qWing=30.000000),
                             broadening=LineBroadening(natural=[RadiativeBroadening(1.155860e+07)],
                                                       elastic=[VdwUnsold([1.0, 1.0]), QuadraticStarkBroadening(1.0), HydrogenLinearStarkBroadening()])),
                   ],
            continua=[
                      HydrogenicContinuum(j=5, i=0, alpha0=6.152000e-22, minWavelength=22.794000, NlambdaGen=20),
                      HydrogenicContinuum(j=5, i=1, alpha0=1.379000e-21, minWavelength=91.176000, NlambdaGen=20),
                      HydrogenicContinuum(j=5, i=2, alpha0=2.149000e-21, minWavelength=205.147000, NlambdaGen=20),
                      HydrogenicContinuum(j=5, i=3, alpha0=2.923000e-21, minWavelength=364.705000, NlambdaGen=20),
                      HydrogenicContinuum(j=5, i=4, alpha0=3.699000e-21, minWavelength=569.852000, NlambdaGen=20),
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

