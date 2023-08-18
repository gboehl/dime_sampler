#!/bin/python
# -*- coding: utf-8 -*-

import os
import emcee
import numpy as np
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import logit, expit
from .dists import inv_gamma_spec, InvGammaDynare


def parse_yaml(mfile):
    """parse from yaml file"""
    import yaml

    f = open(mfile)
    mtxt = f.read()
    f.close()

    # get dict
    return yaml.safe_load(mtxt)


def get_prior(prior, verbose=False):
    """Compile prior-related computational objects from a list of priors.
    """

    prior_lst = ()
    initv, lb, ub = [], [], []
    funcs_con, funcs_re = (), ()  # prior-to-sampler, sampler-to-prior

    snorm = ss.norm()

    if verbose:
        print("Adding parameters to the prior distribution...")

    for pp in prior:

        dist = prior[str(pp)]

        if len(dist) == 3:
            initv.append(None)
            lb.append(None)
            ub.append(None)
            ptype, pmean, pstdd = dist
        elif len(dist) == 6:
            initv.append(eval(str(dist[0])))
            lb.append(dist[1])
            ub.append(dist[2])
            ptype, pmean, pstdd = dist[3:]
        else:
            raise NotImplementedError(
                "Prior specification must either be 3 or 6 inputs but is %s." % pp)

        # simply make use of frozen distributions
        if str(ptype) == "uniform":
            ndist = ss.uniform(loc=pmean, scale=pstdd - pmean)

        elif str(ptype) == "normal":
            ndist = ss.norm(loc=pmean, scale=pstdd)

        elif str(ptype) == "gamma":
            b = pstdd ** 2 / pmean
            a = pmean / b
            ndist = ss.gamma(a, scale=b)

        elif str(ptype) == "beta":
            a = (1 - pmean) * pmean ** 2 / pstdd ** 2 - pmean
            b = a * (1 / pmean - 1)
            ndist = ss.beta(a=a, b=b)

        elif str(ptype) == "inv_gamma":

            def targf(x):
                y0 = ss.invgamma(x[0], scale=x[1]).std() - pstdd
                y1 = ss.invgamma(x[0], scale=x[1]).mean() - pmean
                return np.array([y0, y1])

            ig_res = so.root(targf, np.array([4, 4]), method="lm")

            if ig_res["success"] and np.allclose(targf(ig_res["x"]), 0):
                ndist = ss.invgamma(ig_res["x"][0], scale=ig_res["x"][1])
            else:
                raise ValueError(
                    f"Can not find inverse gamma distribution with mean {pmean} and std {pstdd}.")

        elif str(ptype) == "inv_gamma_dynare":
            s, nu = inv_gamma_spec(pmean, pstdd)
            ndist = InvGammaDynare()(s, nu)

        else:
            raise NotImplementedError(
                f" Distribution {ptype} not implemented.")

        prior_lst += ndist,
        if str(ptype) in ('gamma', 'inv_gamma', 'inv_gamma_dynare'):
            funcs_re += np.log,
            funcs_con += np.exp,
        elif str(ptype) == 'beta':
            funcs_re += logit,
            funcs_con += expit,
        else:
            funcs_re += lambda x: x,
            funcs_con += lambda x: x,

        if verbose:
            if len(dist) == 3:
                print(
                    "   - %s as %s with mean %s and std/df %s"
                    % (pp, ptype, pmean, pstdd)
                )
            if len(dist) == 6:
                print(
                    "   - %s as %s (%s, %s). Init @ %s, with bounds (%s, %s)"
                    % (pp, ptype, pmean, pstdd, dist[0], dist[1], dist[2])
                )

    return list(prior_lst), get_log_prior(prior_lst), get_bijective_prior_transformation(funcs_con, funcs_re), initv, (lb, ub)


def get_log_prior(frozen_prior):
    """Get the log-prior function.
    """

    def log_prior(par):

        prior = 0
        for i, pl in enumerate(frozen_prior):
            prior += pl.logpdf(par[i])

        return prior

    return log_prior


def get_bijective_prior_transformation(funcs_con, funcs_re):
    """Get the bijective prior transformation function.
    """

    def bijective_prior_transformation(x, sampler_to_prior=True):

        x = np.array(x)
        res = x.copy()

        for i in range(x.shape[-1]):
            res[..., i] = funcs_con[i](
                x[..., i]) if sampler_to_prior else funcs_re[i](x[..., i])

        return res

    return bijective_prior_transformation


def find_mode_simple(lprob, init, frozen_prior, sd=True, verbose=False, **kwargs):

    def objective(x):

        ll = -lprob(x) - log_prior(x, frozen_prior)

        if verbose:
            print(-ll)

        return ll

    if not 'method' in kwargs:
        kwargs['method'] = 'Nelder-Mead'

    # minimize objective
    result = so.minimize(objective, init, **kwargs)

    # Compute standard deviation if required
    if sd:
        H, nfev_total = hessian(objective, result.x,
                                nfev=result.nfev, f_x0=result.fun)
        Hinv = np.linalg.inv(H)
        x_sd = np.sqrt(np.diagonal(Hinv))
    else:
        nfev_total = result.nfev
        x_sd = np.zeros_like(result.x)

    return result, x_sd, nfev_total


def save_to_backend(backend, content):

    with backend.open("a") as f:
        g = f[backend.name]
        g["keys"] = list(content.keys())
        for key in content:
            g[key] = content[key]

    return


def load_backend(backend):
    """just a shortcut"""
    reader = emcee.backends.HDFBackend(backend, read_only=True)

    storage_dict = {}
    with reader.open() as f:

        g = f[reader.name]
        if 'keys' in g:
            keys = g["keys"][...]

            for key in keys:
                exec('reader.%s = g[key][...]' % key.decode())

    return reader


def remove_backend(backend):
    """just a shortcut"""
    os.remove(backend)
    return


def map2arr(iterator):
    """Function to cast result from `map` to a tuple of stacked results

    By default, this returns numpy arrays. Automatically checks if the map object is a tuple, and if not, just one object is returned (instead of a tuple). Be warned, this does not work if the result of interest of the mapped function is a single tuple.

    Parameters
    ----------
    iterator : iter
        the iterator returning from `map`

    Returns
    -------
    numpy array (optional: list)
    """

    res = ()
    mode = 0

    for obj in iterator:

        if not mode:
            for entry in obj:
                res = res + ([entry],)
            mode = 1

        else:
            for no, entry in enumerate(obj):
                res[no].append(entry)

    return tuple(np.array(tupo) for tupo in res)


rm_backend = remove_backend
