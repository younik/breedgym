from gym.vector.vector_env import VectorEnvWrapper
from gym import spaces
import jax
import jax.numpy as jnp
from functools import partial


class SelectionValues(VectorEnvWrapper):

    def __init__(self, vec_env, k):
        super().__init__(vec_env)

        self.k = k
        if self.k * (self.k - 1) / 2 > self.individual_per_gen:
            raise ValueError("Invalid value for k. ",
                             "Every diallel pair must have at least a child."
                             )

        self.single_action_space = spaces.Box(
            -1, 1, shape=(self.individual_per_gen,)
        )

        self.action_space = spaces.Box(
            -1, 1, shape=(self.n_envs, self.individual_per_gen)
        )

    @partial(jax.vmap, in_axes=(None, 0))
    def _convert_actions(self, action):
        _, best_pop = jax.lax.top_k(action, self.k)
        diallel_indices = self.simulator._diallel_indices(best_pop)
        n_offspring = jnp.ceil(self.individual_per_gen / len(diallel_indices))
        parents_indices = jnp.repeat(diallel_indices, int(n_offspring), axis=0)
        return parents_indices[:self.individual_per_gen]

    def step_async(self, actions):
        super().step_async(self._convert_actions(actions))
