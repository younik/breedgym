from functools import partial
from math import prod
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chromax.typing import Population
from gymnasium import spaces
from gymnasium.experimental.vector import VectorWrapper
from jaxtyping import Array, Bool, Float, Int

from breedgym.vector import VecBreedGym


class SelectionScores(VectorWrapper):
    def __init__(
        self,
        vec_env: Optional[VecBreedGym] = None,
        k: Optional[int] = None,
        n_crosses: Optional[int] = None,
        **kwargs,
    ):
        if vec_env is None:
            vec_env = VecBreedGym(**kwargs)
        super().__init__(vec_env)

        if k is None:
            k = self.individual_per_gen // 10
        elif k > self.individual_per_gen:
            raise ValueError(
                f"Cannot select {k} best from a population of ",
                f"{self.individual_per_gen} individuals",
            )
        self.k = k

        max_crosses = self.k * (self.k - 1) // 2
        if n_crosses is None:
            n_crosses = max_crosses
        elif n_crosses > max_crosses:
            raise ValueError(
                "Incompatible value for k and n_crosses. ",
                f"With k={k}, the maximum number of crosses is {max_crosses}",
            )
        self.n_crosses = n_crosses

        if self.n_crosses > self.individual_per_gen:
            raise ValueError(
                "Invalid combination for k and n_crosses. ",
                f"Resulting population size will be {n_crosses} ",
                f"that is grater than {self.individual_per_gen}",
            )

        self.single_action_space = spaces.Box(
            -1e5, 1e5, shape=(self.individual_per_gen,)
        )

        self.action_space = spaces.Box(
            -1e5, 1e5, shape=(self.num_envs, self.individual_per_gen)
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _convert_actions(
        self, action: Float[Array, "n"], random_key: jax.random.PRNGKeyArray
    ) -> Int[Array, "n 2"]:
        _, best_pop = jax.lax.top_k(action, self.k)
        diallel_indices = self.simulator._diallel_indices(best_pop)

        random_select_idx = jax.random.choice(
            random_key, len(diallel_indices), shape=(self.n_crosses,), replace=False
        )
        cross_indices = diallel_indices[random_select_idx]

        n_offspring = jnp.ceil(self.individual_per_gen / self.n_crosses)
        return jnp.repeat(
            cross_indices,
            int(n_offspring),
            axis=0,
            total_repeat_length=self.individual_per_gen,
        )

    def step(self, actions: Float[Array, "envs n"]):
        random_keys = jax.random.split(self.random_key, num=self.num_envs + 1)
        self.env.random_key = random_keys[0]
        low_level_actions = self._convert_actions(actions, random_keys[1:])
        return super().step(low_level_actions)


class PairScores(VectorWrapper):
    def __init__(self, vec_env: Optional[VecBreedGym] = None, **kwargs):
        if vec_env is None:
            vec_env = VecBreedGym(**kwargs)
        super().__init__(vec_env)

        self.n_crosses = self.individual_per_gen
        action_shape = self.n_crosses, self.n_crosses
        self.single_action_space = spaces.Box(-1e5, 1e5, shape=action_shape)

        self.action_space = spaces.Box(-1e5, 1e5, shape=(self.num_envs, *action_shape))

    @partial(jax.vmap, in_axes=(None, 0))
    def _convert_actions(self, action: Float[Array, "n n"]) -> Int[Array, "n 2"]:
        best_values, best_crosses = jax.lax.top_k(action.flatten(), self.n_crosses)
        offspring_per_cross = jax.nn.softmax(best_values) * self.n_crosses
        row_indices = best_crosses // self.n_crosses
        col_indices = best_crosses % self.n_crosses
        cross_indices = jnp.stack((row_indices, col_indices), axis=1)
        return jnp.repeat(
            cross_indices,
            jnp.ceil(offspring_per_cross).astype(jnp.int32),
            axis=0,
            total_repeat_length=self.n_crosses,
        )

    def step(
        self, actions: Float[Array, "envs n n"]
    ) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict,
    ]:
        low_level_actions = self._convert_actions(actions)
        return super().step(low_level_actions)


class RavelIndex(VectorWrapper):
    def __init__(self, vec_env: VecBreedGym):
        super().__init__(vec_env)
        self.action_shape = self.single_action_space.shape
        n_elems = prod(self.action_shape)
        n_vec = jnp.full((self.individual_per_gen,), n_elems)
        self.single_action_space = spaces.MultiDiscrete(n_vec)
        self.action_space = spaces.MultiDiscrete(
            jnp.broadcast_to(n_vec[None, ...], (self.num_envs, *n_vec.shape))
        )

    @partial(jax.vmap, in_axes=(None, 0))
    def _convert_actions(self, action: Float[Array, "n"]) -> Int[Array, "n 2"]:
        indices = jnp.unravel_index(action, self.action_shape)
        return jnp.stack(indices, axis=1)

    def step(
        self, actions: Float[Array, "envs n"]
    ) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict,
    ]:
        actions = self._convert_actions(actions)
        return super().step(actions)
