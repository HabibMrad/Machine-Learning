#\!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation of gradient boosting technique with one model."""

import warnings

import numpy as np
from scipy.special import expit as sigmoid

from optimizers import adam


def find_gamma(initial_gamma, loss, n_steps: int):
    """Finds gamma after n_steps minimzation of loss function.

    :param initial_gamma: Starting point for gradient optimization
    :param loss: Loss function to optimize (object containing taylor method).
    Taylor methods has to return two values:
    a) function at point
    b) gradient at point
    :param n_steps: number of steps for optimizer
    """
    initial_gamma = np.array([initial_gamma])  # GammaLoss oczekuje tablicy
    optimizer = adam(
        f=loss,
        starting_point=initial_gamma,
        learning_rate=.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8)
    gamma = initial_gamma
    for _ in range(n_steps):
        gamma, _, _ = next(optimizer)
    return gamma[0]


def binary_crossentropy_pseudo_residuals(y_true, logits):
    """Pseudo residuals between true label values and predicted logits.

    Arguments need equal shape

    :param y_true: True values (np.array)
    :param logits: predicted logit values (np.array)
    """
    return y_true - sigmoid(logits)


class GammaLoss:
    """Functor containing derivative (used in find_gamma)."""

    def __init__(self, y_true, logits, r):
        """__init__

        :param y_true: True values (np.array)
        :param logits: predicted logit values (np.array)
        :param r: predictions of last model on original dataset
        """
        self.y_true = y_true
        self.logits = logits
        self.r = r

    def taylor(self, gamma):
        """Calculates derivative of loss function w.r.t. gamma

        :param gamma: gamma parameter
        :returns: function value for specific gamma, derivative of function
        """
        return \
            -np.mean(self.y_true * np.log(sigmoid(gamma * self.logits +
                                                  gamma * self.r)) +
                     (1 - self.y_true) * np.log(sigmoid((1 - gamma) *
                                                        self.logits +
                                                        gamma * self.r))),\
            -np.mean(self.r * (self.y_true -
                               sigmoid(self.logits + self.r * gamma)))


class GradientBoostingClassifier:
    """Class implementing gradient boosting."""

    def __init__(self,
                 X,
                 y,
                 n_models,
                 model_cls,
                 train_fraction: float = .7,
                 initial_gamma: float = 1.,
                 gamma_n_steps: int = 200,
                 seed=43):
        """__init__

        :param X: Dataset to perform gradient boosting on. np.array, shaped
        (observations, features)
        :param y: Array containing appropriate labels. np.array, shaped
        (observations,)
        :param n_models: Number of models to use for gradient boosting (int)
        :param train_fraction: Float describing how many training examples
        should be taken for each model (e.g. 0.7 will use 70% of dataset).
        Randomized samples for each model
        :param initial_gamma: Starting parameter gamma for boosting (int)
        :param gamma_n_steps: Number of steps to perform for boosting (int)
        :param seed: Seed for random generator
        """

        np.random.seed(seed)

        # CALCULATE NUMBER OF EXAMPLES USED FOR TRAINING
        subset = int(X.shape[0] * train_fraction)

        # CONCATENATE X AND Y TO SHUFFLE EVERYTHING TOGETHER IN LOOP BELOW
        dataset = np.c_[X, y]

        # INITIALIZE LIST OF MODELS AND GAMMAS
        self.models = []
        self.gammas = []

        for _ in range(n_models):
            np.random.shuffle(dataset)
            predicted_logits = self.predict_logits(dataset[:subset, :-1])
            pseudo_residuals = binary_crossentropy_pseudo_residuals(
                dataset[:subset, -1], predicted_logits)
            self.models.append(
                model_cls(dataset[:subset, :-1], pseudo_residuals))

            last_predictions = self.models[-1].predict(dataset[:subset, :-1])
            self.gammas.append(
                find_gamma(initial_gamma,
                           GammaLoss(dataset[:subset, -1], predicted_logits,
                                     last_predictions), gamma_n_steps))

    def predict_logits(self, X, step: int = None):
        """Predicts logits accordingly to dataset X

        :param X: Dataset of shape (observations, features) [np.array]
        :param step: Number of models used for prediction (int)
        :returns: Predicted logits corresponding to dataset
        of shape (observations,) [np.array]
        """
        if step is None:
            step = len(self.models)

        return np.zeros(X.shape[0]) + np.sum(
            np.array([
                gamma * model.predict(X)
                for gamma, model in zip(self.gammas[:step], self.models[:step])
            ]),
            axis=0)

    def predict(self, X, step=None):
        """predict

        :param X: Dataset of shape (observations, features) [np.array]
        :param step: Number of models used for prediction [int]
        :returns: Labels containing predictions of the model of shape
        (observations,) [np.array]
        """
        return (self.predict_logits(X, step) >= 0.).astype(np.int)
