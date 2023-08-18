#!/bin/python
# -*- coding: utf-8 -*-

import warnings
import emcee
import numpy as np
import pandas as pd
import scipy.stats as ss

try:
    from grgrlib.njitted import mvn_logpdf as logpdf
except ModuleNotFoundError:
    def logpdf(drv, cmean, ccov):
        return ss.multivariate_normal(mean=cmean, cov=ccov).logpdf(drv)


def mdd_laplace(chain, lprobs, calc_hess=False):
    """Approximate the marginal data density useing the LaPlace method."""

    if chain.ndim > 2:
        chain = chain.reshape(-1, chain.shape[2])

    lprobs = lprobs.flatten()

    mode_x = chain[lprobs.argmax()]

    if calc_hess:

        try:
            from numdifftools import Hessian
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`mdd_laplace` with `calc_hess=True` requires the numdifftools package to be installed.")

        np.warnings.filterwarnings("ignore")
        hh = Hessian(func)(mode_x)
        np.warnings.filterwarnings("default")

        if np.isnan(hh).any():
            raise ValueError(
                "[mdd:]".ljust(15, " ")
                + "Option `hess` is experimental and did not return a usable hessian matrix."
            )

        inv_hess = np.linalg.inv(hh)

    else:
        inv_hess = np.cov(chain.T)

    ndim = chain.shape[-1]
    log_det_inv_hess = np.log(np.linalg.det(inv_hess))
    mdd = 0.5 * ndim * np.log(2 * np.pi) + 0.5 * \
        log_det_inv_hess + lprobs.max()

    return mdd


def mdd_harmonic_mean(chain, lprobs, pool=None, alpha=0.05, verbose=False, debug=False):
    """Approximate the marginal data density useing modified harmonic mean."""

    if chain.ndim > 2:
        chain = chain.reshape(-1, chain.shape[2])

    lprobs = lprobs.flatten()

    cmean = chain.mean(axis=0)
    ccov = np.cov(chain.T)
    cicov = np.linalg.inv(ccov)

    nsamples = chain.shape[0]

    def runner(chunk):

        res = np.empty_like(chunk)
        wrapper = tqdm.tqdm if verbose else (lambda x, **kwarg: x)

        for i in wrapper(range(len(chunk))):

            drv = chain[i]
            drl = lprobs[i]

            if (drv - cmean) @ cicov @ (drv - cmean) < ss.chi2.ppf(
                1 - alpha, df=chain.shape[-1]
            ):
                res[i] = logpdf(drv, cmean, ccov) - drl
            else:
                res[i] = -np.inf

        return res

    if not debug and pool is not None:
        nbatches = pool.ncpus
        batches = pool.imap(runner, np.array_split(chain, nbatches))
        mls = np.vstack(list(batches))
    else:
        mls = runner(chain)

    maxllike = np.max(mls)  # for numeric stability
    imdd = np.log(np.mean(np.exp(mls - maxllike))) + maxllike

    return -imdd


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        # raise ValueError('Too few elements for interval calculation')
        warnings.warn("Too few elements for interval calculation.")

        return None, None

    else:
        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]

        return hdi_min, hdi_max


def mc_error(x):
    means = np.mean(x, 0)
    return np.std(means) / np.sqrt(x.shape[0])


def _hpd_df(x, alpha):

    cnames = [
        "hpd_{0:g}".format(100 * alpha / 2),
        "hpd_{0:g}".format(100 * (1 - alpha / 2)),
    ]

    sx = np.sort(x.flatten())
    hpd_vals = np.array(calc_min_interval(sx, alpha)).reshape(1, -1)

    return pd.DataFrame(hpd_vals, columns=cnames)


def gelman_rubin(sample):
    """The Gelman-Rubin (1992) statistics.
    """

    ndraws, nchains, npars = sample.shape

    chain_mean = sample.mean(axis=0)
    between_chain_var = ndraws*np.var(chain_mean, axis=0, ddof=1)
    within_chain_var = np.var(sample, axis=0, ddof=1)
    mean_variance = 1/nchains*np.sum(within_chain_var, axis=0)

    return 1 - 1/ndraws + 1/ndraws*between_chain_var/mean_variance


def summary(priors, store, pmode=None, bounds=None, alpha=0.1, top=None, show_prior=True):
    # inspired by pymc3 because it looks really nice

    if bounds is not None or isinstance(store, tuple):
        xs, fs, ns = store
        ns = ns.squeeze()
        fas = (-fs[:, 0]).argsort()
        xs = xs[fas]
        fs = fs.squeeze()[fas]

    f_prs = [
        lambda x: pd.Series(x, name="distribution"),
        lambda x: pd.Series(x, name="pst_mean"),
        lambda x: pd.Series(x, name="sd/df"),
    ]

    f_bnd = [
        lambda x: pd.Series(x, name="lbound"),
        lambda x: pd.Series(x, name="ubound"),
    ]

    def mode_func(x, n):
        return pmode[n] if pmode is not None else mode(x.flatten())

    funcs = [
        lambda x, n: pd.Series(np.mean(x), name="mean"),
        lambda x, n: pd.Series(np.std(x), name="sd"),
        lambda x, n: pd.Series(
            mode_func(x, n), name="mode" if pmode is not None else "marg. mode"
        ),
        lambda x, n: _hpd_df(x, alpha),
        lambda x, n: pd.Series(mc_error(x), name="error"),
    ]

    var_dfs = []
    for i, var in enumerate(priors):

        lst = []
        if show_prior:
            prior = priors[var]
            if len(prior) > 3:
                prior = prior[-3:]
            [lst.append(f(prior[j])) for j, f in enumerate(f_prs)]
            if bounds is not None:
                [lst.append(f(np.array(bounds).T[i][j]))
                 for j, f in enumerate(f_bnd)]

        if bounds is not None:
            [lst.append(pd.Series(s[i], name=n))
             for s, n in zip(xs[:top], ns[:top])]
        else:
            vals = store[:, :, i]
            [lst.append(f(vals, i)) for f in funcs]
        var_df = pd.concat(lst, axis=1)
        var_df.index = [var]
        var_dfs.append(var_df)

    if bounds is not None:

        lst = []

        if show_prior:
            [lst.append(f("")) for j, f in enumerate(f_prs)]
            if bounds is not None:
                [lst.append(f("")) for j, f in enumerate(f_bnd)]

        [lst.append(pd.Series(s, name=n)) for s, n in zip(fs[:top], ns[:top])]
        var_df = pd.concat(lst, axis=1)
        var_df.index = ["loglike"]
        var_dfs.append(var_df)

    dforg = pd.concat(var_dfs, axis=0, sort=False)

    return dforg


# a wrapper without warning
def integrated_time(x): return emcee.autocorr.integrated_time(x, tol=0)
