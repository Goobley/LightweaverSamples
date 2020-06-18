from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import matplotlib.pyplot as plt
import time
import numpy as np
from lightweaver.utils import NgOptions, get_default_molecule_path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from contextlib import redirect_stdout
import os
import sys

wave = np.linspace(853.9444, 854.9444, 1001)

class PickleMe:
    def __getstate__(self):
        print('I\'m being pickled!', file=sys.stderr)
        return {}

    def __setstate__(self, state):
        print('Being unpickled!', file=sys.stderr)

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

def compute_solution(ctx, *args):
    iterate_ctx(ctx)
    ctx.pops.update_lte_atoms_Hmin_pops(ctx.arguments['atmos'])
    Iwave = ctx.compute_rays(wave, 1.0, stokes=False)
    return Iwave

def make_fal_context(velShift=0.0):
    atmos = Falc82()
    atmos.vlos[:] = velShift
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)

    ctx = lw.Context(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), Nthreads=1)
    return ctx

def shush(fn, *args, **kwargs):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            return fn(*args, **kwargs)

numProcesses = min(cpu_count(), 16)
vels = np.linspace(0.0, 30e3, numProcesses).tolist()
def test_process_pool():
    singleCtx = make_fal_context(0.0)
    picklee = PickleMe()
    singleStart = time.time()
    print('Doing single context')
    # NOTE(cmo): picklee won't print here
    Iwave0 = shush(compute_solution, singleCtx, picklee)
    singleEnd = time.time()

    # NOTE(cmo): This is a slow way of doing ProcessPool as the Contexts are
    # first made on one thread, and then pickled and depickled on the other end
    # (as proven by picklee). In most cases the Context can be made on the
    # required process, or shared in some way by pre-existing on the main
    # process and being available through `spawn(2)` -- although this can be
    # problematic too if Nthreads != 0 on the Context.
    # This demonstrates that pickling works just fine!
    ctxs = [make_fal_context(v) for v in vels]
    print('Doing contexts in parallel')
    multiStart = time.time()
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(shush, compute_solution, c, picklee) for c in ctxs]

        for f in tqdm(as_completed(futures), total=len(futures)):
            pass

    Iwaves = [f.result() for f in futures]
    multiEnd = time.time()

    if not np.allclose(Iwave0, Iwaves[0]):
        raise ValueError('Single and Distributed don\'t match!!')

    sDuration = singleEnd - singleStart
    mDuration = multiEnd - multiStart
    print('Time for single ctx: %.3g s' % sDuration)
    print('Time for ProcessPool ctxs (%d): %.3g s' % (len(ctxs), mDuration))
    print('Ratio (max possible = 1.0, or ~0.5 with HT): %.4g' % (sDuration / mDuration))

    return Iwaves

if __name__ == '__main__':
    Iwaves = test_process_pool()

    plt.ion()
    for i, v in enumerate(vels):
        plt.plot(wave, Iwaves[i], label='%g km/s' % v)
    plt.legend()