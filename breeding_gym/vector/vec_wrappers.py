from gym.vector.vector_env import VectorEnvWrapper
from gym import spaces
import jax
import jax.numpy as jnp
from functools import partial


class SelectionValues(VectorEnvWrapper):

    def __init__(self, vec_env, k, n_crosses=None):
        super().__init__(vec_env)

        if k > self.individual_per_gen:
            raise ValueError(f"Cannot select {k} best from a population of ",
                             f"{self.individual_per_gen} individuals"
                             )
        self.k = k

        max_crosses = self.k * (self.k - 1) // 2
        if n_crosses is None:
            n_crosses = max_crosses
        elif n_crosses > max_crosses:
            raise ValueError(
                "Incompatible value for k and n_crosses. ",
                f"With k={k}, the maximum number of crosses is {max_crosses}"
            )
        self.n_crosses = n_crosses

        if self.n_crosses > self.individual_per_gen:
            raise ValueError("Invalid combination for k and n_crosses. ",
                             f"Resulting population size will be {n_crosses} ",
                             f"that is grater than {self.individual_per_gen}"
                             )

        self.single_action_space = spaces.Box(
            -1e5, 1e5, shape=(self.individual_per_gen,)
        )

        self.action_space = spaces.Box(
            -1e5, 1e5, shape=(self.n_envs, self.individual_per_gen)
        )

    @partial(jax.vmap, in_axes=(None, 0))
    def _convert_actions(self, action):
        _, best_pop = jax.lax.top_k(action, self.k)
        diallel_indices = self.simulator._diallel_indices(best_pop)

        self.random_key, split_key = jax.random.split(self.random_key)
        random_select_idx = jax.random.choice(
            split_key,
            len(diallel_indices),
            shape=(self.n_crosses,),
            replace=False
        )
        cross_indices = diallel_indices[random_select_idx]

        n_offspring = jnp.ceil(self.individual_per_gen / self.n_crosses)
        parents_indices = jnp.repeat(cross_indices, int(n_offspring), axis=0)
        return parents_indices[:self.individual_per_gen]

    def step_async(self, actions):
        super().step_async(self._convert_actions(actions))
