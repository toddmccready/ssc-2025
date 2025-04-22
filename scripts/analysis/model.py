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

# Construct network architecture
n_features: int = 159
d_inner: int = 100

key: Key = jr.PRNGKey(0)
mu_arch = [
    eqx.nn.LayerNorm(n_features),
    eqx.nn.MLP(n_features, 'scalar', d_inner, 5, key = key),
]
sigma_arch = [
    eqx.nn.LayerNorm(n_features),
    eqx.nn.MLP(n_features, 'scalar', d_inner, 5, key = key, final_activation = jnp.exp),
]

class UncensoredModel(Model):
    mu_mapping: Parameter[eqx.nn.Sequential]
    sigma_mapping: Parameter[eqx.nn.Sequential]

    def __init__(self):
        self.mu_mapping = Parameter(
            eqx.nn.Sequential(
                mu_arch
            )
        )
        self.sigma_mapping = Parameter(
            eqx.nn.Sequential(
                sigma_arch
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

    def __init__(self):
        self.mu_mapping = Parameter(
            eqx.nn.Sequential(
                mu_arch
            )
        )
        self.sigma_mapping = Parameter(
            eqx.nn.Sequential(
                sigma_arch
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
