#!/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from .moves import DIMEMove
from .sampler import run_mcmc
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp
from emcee.tests.integration.test_proposal import _test_normal, _test_uniform

filepath = os.path.dirname(__file__)


def _create_test_func(ndim, weight, distance, cov_scale):

    cov = np.eye(ndim)*cov_scale
    mean = np.zeros(ndim)
    mean[0] = distance

    lw0 = np.log(weight[0])
    lw1 = np.log(weight[1])
    lw2 = np.log(1-weight[0]-weight[1])

    dist = multivariate_normal(np.zeros(ndim), cov)

    def log_prob(p):
        return logsumexp((lw0 + dist.logpdf(p + mean), lw1 + dist.logpdf(p), lw2 + dist.logpdf(p - mean)))

    return log_prob


def _marginal_pdf_test_func(x, cov_scale, m, weight):

    normal = norm(scale=np.sqrt(cov_scale))

    return weight[0]*normal.pdf(x+m) + weight[1]*normal.pdf(x) + (1-weight[0]-weight[1])*normal.pdf(x-m)


def test_verbose(seed=1234):

    np.random.seed(seed)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = (0.33, 0.1)
    ndim = 35

    log_prob = _create_test_func(ndim, weight, m, cov_scale)

    nchain = ndim*5
    niter = 300

    initmean = np.zeros(ndim)
    initcov = np.eye(ndim)*np.sqrt(2)
    initchain = multivariate_normal(
        mean=initmean, cov=initcov, seed=seed).rvs(nchain)

    moves = DIMEMove(aimh_prob=.1, df_proposal_dist=10)
    # test if printing works
    _ = run_mcmc(log_prob, 2, p0=initchain, moves=moves, verbose=True)
    assert True


def test_example(create=False, seed=1234):

    np.random.seed(seed)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = (0.33, 0.1)
    ndim = 35

    log_prob = _create_test_func(ndim, weight, m, cov_scale)

    nchain = ndim*5
    niter = 300

    initmean = np.zeros(ndim)
    initcov = np.eye(ndim)*np.sqrt(2)
    initchain = multivariate_normal(
        mean=initmean, cov=initcov, seed=seed).rvs(nchain)

    moves = DIMEMove(aimh_prob=.1, df_proposal_dist=10)
    # test silently
    sampler = run_mcmc(log_prob, niter, p0=initchain,
                       moves=moves, verbose=create)
    chain = sampler.get_chain()

    path = os.path.join(filepath, "test_storage", "median.npy")
    median = np.median(chain[-int(niter/3):, :, 0])

    if create:
        np.save(path, median)
        print(f'Test file updated at {path} with {median}')
    else:
        test_median = np.load(path)
        np.testing.assert_allclose(median, test_median, atol=1e-4)


def test_normal_dime(**kwargs):
    _test_normal(DIMEMove(), **kwargs)


def test_uniform_dime(**kwargs):
    _test_uniform(DIMEMove(), **kwargs)
