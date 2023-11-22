from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.experimental.vector import VectorWrapper

from breedgym.vector import VecBreedGym


class WheatBreedGym(VectorWrapper):
    def __init__(
        self, vec_env: VecBreedGym, n_lines=200, plant_per_line=100, k_per_line=5
    ):
        super().__init__(vec_env)
        self.n_lines = n_lines
        self.plant_per_line = plant_per_line
        self.k_per_line = k_per_line
        action_shape = self.n_lines, self.n_lines
        self.single_action_space = spaces.Box(-1e5, 1e5, shape=action_shape)

        self.action_space = spaces.Box(-1e5, 1e5, shape=(self.num_envs, *action_shape))

    @partial(jax.vmap, in_axes=(None, 0))
    def _convert_actions(self, action):
        best_values, best_crosses = jax.lax.top_k(action.flatten(), self.n_lines)
        offspring_per_cross = jax.nn.softmax(best_values) * self.n_lines
        row_indices = best_crosses // self.n_lines
        col_indices = best_crosses % self.n_lines
        cross_indices = jnp.stack((row_indices, col_indices), axis=1)
        return jnp.repeat(
            cross_indices,
            jnp.ceil(offspring_per_cross).astype(jnp.int32),
            axis=0,
            total_repeat_length=self.n_lines,
        )

    @partial(jax.vmap, in_axes=(None, 0))
    def _double_haploid(self, pop):
        return self.simulator.double_haploid(pop, n_offspring=self.plant_per_line)

    @partial(jax.vmap, in_axes=(None, 0))
    def _select_line(self, pop):
        return self.simulator.select(pop, k=self.k_per_line)

    def step(self, actions):
        actions = self._convert_actions(actions)
        arange_envs = np.arange(self.num_envs)[:, None, None]
        parents = self.populations[arange_envs, actions]
        pop = self.cross(parents)
        assert pop.shape[1] == self.n_lines
        pop = self._double_haploid(pop)
        pop = self._select_line(pop)
        pop = pop.reshape(pop.shape[0], pop.shape[1] * pop.shape[2], *pop.shape[3:])
        self.env.populations = self.simulator.select(pop, k=self.individual_per_gen)

        self.env.step_idx += 1
        infos = self.get_info()
        done = self.env.step_idx == self.num_generations
        if self.reward_shaping or done:
            rews = np.max(infos["GEBV"], axis=(1, 2))
            rews = np.asarray(rews)
        else:
            rews = np.zeros(self.num_envs)

        if done and self.autoreset:
            self.reset()

        terminated = np.full(self.num_envs, False)
        truncated = np.full(self.num_envs, done)
        return self.populations, rews, terminated, truncated, infos
