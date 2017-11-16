# coding=utf-8
"""Moudle with BayesianDensityEstimator implementation.

It can return:
1. Likelihood for observations (normal, not log)
2. Current posterior (as property)
3. PPD for data points
4. Current prior

"""

import numpy as np


class BayesianDensityEstimator:
    """BayesianDensityEstimator.

    Implements Bayesian Density Estimation using multiple distributions
    parametrized by given params (see __init__)

    """

    def __init__(self, distr_cls, params, prior):
        """__init__. Creates distributions with given parameters, initializes
        posterior and prior to prior.

        :param params: parameters for each distribution
        :param prior: Bayes prior

        """
        self.distributions = [distr_cls(param) for param in params]
        self.prior = prior
        self.posterior = prior

    def likelihood(self, observations):
        """likelihood. Implementation of Bayes likelihood for len(params)
        distributions.

        :param observations: numpy array shaped(observations, traits)
        shaped(observations,length_of_traits_vector)
        :returns: np.array shape(len(distributions), ) with joined likelihood
        of all observations

        """
        return np.array([
            np.prod(distribution.pdf(observations))
            for distribution in self.distributions
        ])

    def ppd(self, X):
        """ppd. Return Posterior Predictive Distribution for data samples X.

        :param X: numpy array of shape(x, traits)
        :returns: numpy array of ppd for each x

        """
        return np.sum(
            np.array(
                [distribution.pdf(X)
                 for distribution in self.distributions]) * self.posterior,
            axis=1)

    def observe(self, observations):
        """observe. Calculates posterior and updates prior using likelihood.

        Values are assigned to self.posterior and self.prior and nothing is
        returned.

        :param observations: numpy array shaped(observations, traits)
        :returns: None

        """
        # NORMALIZUJ POSTERIOR
        likelihood = self.likelihood(observations)
        self.posterior = likelihood * self.prior
        self.prior = self.posterior
