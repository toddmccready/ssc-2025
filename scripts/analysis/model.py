from functools import partial
from typing import Dict

import bayinx.dists.censored as cens
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from bayinx import Model, Parameter
from bayinx.dists import posnormal
from jaxtyping import Array, Key, Scalar

n_features: int = 159
d1_inner: int = 512
d2_inner: int = 256


class UncensoredModel(Model):
    mu_mapping: Parameter[eqx.nn.Sequential]
    sigma_mapping: Parameter[eqx.nn.Sequential]

    def __init__(self, key: Key = jr.PRNGKey(0)):
        keys: Key = jr.split(key, 6)
        self.mu_mapping = Parameter(
            eqx.nn.Sequential(
                [
                    eqx.nn.LayerNorm(n_features),
                    eqx.nn.Linear(n_features, d1_inner, key=keys[0]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(d1_inner, d2_inner, key=keys[1]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(d2_inner, "scalar", key=keys[2]),
                ]
            )
        )
        self.sigma_mapping = Parameter(
            eqx.nn.Sequential(
                [
                    eqx.nn.LayerNorm(n_features),
                    eqx.nn.Linear(n_features, d1_inner, key=keys[3]),
                    eqx.nn.Lambda(jax.nn.leaky_relu),
                    eqx.nn.Linear(d1_inner, d2_inner, key=keys[4]),
                    eqx.nn.Lambda(jax.nn.leaky_relu),
                    eqx.nn.Linear(d2_inner, "scalar", key=keys[5]),
                    eqx.nn.Lambda(jnp.exp),
                ]
            )
        )

    @eqx.filter_jit
    @partial(eqx.filter_vmap, in_axes=(None, 1))
    def map_mu(self, x: Array) -> Scalar:
        return self.mu_mapping()(x)

    @eqx.filter_jit
    @partial(eqx.filter_vmap, in_axes=(None, 1))
    def map_sigma(self, x: Array) -> Scalar:
        return self.sigma_mapping()(x)

    @eqx.filter_jit
    def eval(self, data: Dict[str, Array]):
        self, target = self.constrain_params()

        # Compute expected time-to-outcome
        mu: Array = self.map_mu(data["predictors"])
        sigma: Array = self.map_sigma(data["predictors"])

        # Accumulate likelihood
        target += posnormal.logprob(
            x=data["outcome"],
            mu=mu,
            sigma=sigma,
        ).sum()

        return target


class CensoredModel(Model):
    mu_mapping: Parameter[eqx.nn.Sequential]
    sigma_mapping: Parameter[eqx.nn.Sequential]

    def __init__(self, key: Key = jr.PRNGKey(0)):
        keys: Key = jr.split(key, 6)
        self.mu_mapping = Parameter(
            eqx.nn.Sequential(
                [
                    eqx.nn.LayerNorm(n_features),
                    eqx.nn.Linear(n_features, d1_inner, key=keys[0]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(d1_inner, d2_inner, key=keys[1]),
                    eqx.nn.Lambda(jax.nn.relu),
                    eqx.nn.Linear(d2_inner, "scalar", key=keys[2]),
                ]
            )
        )
        self.sigma_mapping = Parameter(
            eqx.nn.Sequential(
                [
                    eqx.nn.LayerNorm(n_features),
                    eqx.nn.Linear(n_features, d1_inner, key=keys[3]),
                    eqx.nn.Lambda(jax.nn.leaky_relu),
                    eqx.nn.Linear(d1_inner, d2_inner, key=keys[4]),
                    eqx.nn.Lambda(jax.nn.leaky_relu),
                    eqx.nn.Linear(d2_inner, "scalar", key=keys[5]),
                    eqx.nn.Lambda(jnp.exp),
                ]
            )
        )

    @eqx.filter_jit
    @partial(eqx.filter_vmap, in_axes=(None, 1))
    def map_mu(self, x: Array) -> Scalar:
        return self.mu_mapping()(x)

    @eqx.filter_jit
    @partial(eqx.filter_vmap, in_axes=(None, 1))
    def map_sigma(self, x: Array) -> Scalar:
        return self.sigma_mapping()(x)

    @eqx.filter_jit
    def eval(self, data: Dict[str, Array]):
        self, target = self.constrain_params()

        # Compute expected time-to-outcome
        mu: Array = self.map_mu(data["predictors"])
        sigma: Array = self.map_sigma(data["predictors"])

        # Accumulate likelihood
        target += cens.posnormal.r.logprob(
            x=data["outcome"],
            mu=mu,
            sigma=sigma,
            censor=data["censoring_time"],
        ).sum()

        return target

@eqx.filter_jit
def get_intervals(model: UncensoredModel | CensoredModel, data: Dict[str, Array]):
    model = model.constrain_params()[0]

    # Compute distribution parameters
    mu = model.map_mu(data["predictors"])
    sigma = model.map_sigma(data["predictors"])

    # Sample from predictive
    predictions = cens.posnormal.r.sample(int(1e4), mu, sigma)

    # Compute quantiles
    quantiles: Array = jnp.stack([data["alpha"] / 2, 1.0 - data["alpha"] / 2], axis=1)

    return jax.vmap(jnp.quantile, in_axes=(None, 0, None))(predictions, quantiles, 0)
