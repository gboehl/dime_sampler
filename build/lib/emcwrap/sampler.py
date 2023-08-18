#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import tqdm
import emcee
import warnings
import numpy as np
from .tools import map2arr
from .stats import summary
from .moves import DIMEMove


def run_mcmc(lprob, nsteps=None, p0=None, moves=None, stopping_weight=None, priors=None, prior_transform=None, backend=None, update_freq=False, resume=False, pool=None, report=None, description=None, temp=1, maintenance_interval=False, verbose=True, **kwargs):
    """Run the emcee sampler.
    """

    if prior_transform is None:
        def prior_transform(x): return x

    if moves is None:
        moves = DIMEMove()

    if isinstance(backend, str):
        backend = emcee.backends.HDFBackend(
            os.path.splitext(backend)[0] + '.h5')

    if resume:
        p0 = backend.get_chain()[-1]

    nwalks, ndim = np.shape(p0)

    sampler = emcee.EnsembleSampler(
        nwalks, ndim, lprob, moves=moves, pool=pool, backend=backend)

    if not verbose:  # verbose means VERY verbose
        warnings.filterwarnings("ignore")

    if verbose > 1:
        report = report or print
    elif verbose:
        pbar = tqdm.tqdm(total=nsteps, unit="sample(s)", dynamic_ncols=True)
        report = report or pbar.write
    else:
        def report(x): return None

    old_tau = np.inf
    old_lls = np.array([-np.inf]*nwalks)
    cumlweight = -np.inf
    cnt = 0

    for result in sampler.sample(p0, iterations=nsteps, **kwargs):

        if hasattr(moves, "cumlweight"):
            state_weight = cumlweight - moves.cumlweight
            cumlweight = moves.cumlweight

            if stopping_weight is not None:
                if state_weight > np.log(1 - stopping_weight):
                    if verbose:
                        print(
                            f"(mcmc:) desired incremental weight of {stopping_weight:.1e} reached. Exiting...")
                    break

        if verbose == 1:
            lls = list(result)[1]
            maf = 1-sum(lls == old_lls)/nwalks
            try:
                maf = f"{maf:3.0%}"
            except BlockingIOError:
                maf = "??"
            desc_str = f"[ll/MAF:{np.max(lls):7.3f}({np.std(lls):1.0e})/{maf}"
            if hasattr(moves, "cumlweight"):
                desc_str += f" | {state_weight:1.0e}"
            desc_str += "]"

            pbar.set_description(desc_str)
            old_lls = lls.copy()

        if cnt and update_freq and not cnt % update_freq:

            prnttup = f"(mcmc:) Summary from last {update_freq} of {cnt} iterations"

            if temp < 1:
                prnttup += f" with temp of {temp * 100:1.6f}"

            if description is not None:
                prnttup += f" ({str(description)})"

            report(prnttup + ':')

            sample = sampler.get_chain()
            lprobs = sampler.get_log_prob(flat=True)
            acfs = sampler.acceptance_fraction

            tau = emcee.autocorr.integrated_time(sample, tol=0, c=10)
            min_tau = np.min(tau).round(2)
            max_tau = np.max(tau).round(2)

            if priors is not None:
                mcmc_summary(
                    chain=prior_transform(sample[-update_freq:]),
                    lprobs=lprobs[-update_freq:],
                    priors=priors,
                    acceptance_fraction=acfs[-update_freq:],
                    out=lambda x: report(str(x)),
                )

            report(
                f"Autocorrelation times are between {min_tau} and {max_tau}."
            )

        if cnt and update_freq and not (cnt + 1) % update_freq:
            sample = sampler.get_chain()
            old_tau = emcee.autocorr.integrated_time(sample, tol=0)

        if verbose == 1:
            pbar.update(1)

        # avoid mem leakage
        if maintenance_interval and cnt and pool and not cnt % maintenance_interval:
            pool.clear()

        cnt += 1

    if verbose == 1:
        pbar.close()
    if pool:
        pool.close()

    if not verbose:
        warnings.filterwarnings("default")

    if backend is None:
        return sampler
    else:
        return backend


def mcmc_summary(
    chain,
    lprobs,
    priors,
    acceptance_fraction=None,
    out=print,
    **args
):

    nchain = chain.reshape(-1, chain.shape[-1])
    lprobs = lprobs.reshape(-1, lprobs.shape[-1])
    mode_x = nchain[lprobs.argmax()]

    res = summary(priors, chain, mode_x, **args)

    out(res.round(3))

    if acceptance_fraction is not None:

        out("Mean acceptance fraction:" +
            str(np.mean(acceptance_fraction).round(3)).rjust(13))

    return res


def get_prior_sample(frozen_prior, nsamples, check_func=False, seed=None, mapper=map, filterwarnings='error', max_attempts=10, debug=False, verbose=True):
    """Get a sample of size `nsamples` from the prior distribution.
    """

    if seed is None:
        seed = np.random.randint(2**31)

    if check_func and not callable(check_func):
        raise Exception('`check_func` must be `False` or a callable')

    def runner(locseed):

        # distribute seed evenly
        np.random.seed((seed + locseed) % 2**31)
        done = False
        no = 0

        while not done:

            no += 1

            with warnings.catch_warnings(record=False):
                try:
                    # set warnings
                    warnings.filterwarnings(filterwarnings)
                    rst = np.random.randint(2 ** 31)  # win explodes with 2**32
                    # draw from prior
                    pdraw = [
                        pl.rvs(random_state=rst + sn)
                        for sn, pl in enumerate(frozen_prior)
                    ]

                    # check if function returns finite results (if provided)
                    if check_func:
                        draw_prob = check_func(pdraw)
                        done = not np.any(np.isinf(draw_prob))
                    else:
                        done = True

                    # let non-finite count as errors
                    if not done:
                        raise ValueError("result not finite")

                # be kind to errors
                except Exception as e:
                    if debug:
                        print(str(e) + f" ({no})")
                    if no >= max_attempts:
                        raise type(e)(
                            str(e) + f" (after {no} unsuccessful attemps).").with_traceback(sys.exc_info()[2])
                    else:
                        pass

        return pdraw, no

    if verbose > 1:
        print("(prior_sample:) Sampling from the pior...")

    wrapper = tqdm.tqdm if verbose < 2 else (lambda x, **kwarg: x)
    pmap_sim = wrapper(mapper(runner, range(nsamples)), total=nsamples)

    draws, nos = map2arr(pmap_sim)

    if verbose and check_func:
        print(
            f"(prior_sample:) Sampling done. Check fails for {100 * (sum(nos) - nsamples) / sum(nos):2.2f}% of the prior.")

    return draws
