from math import sqrt
from gym.vector.vector_env import VectorEnv
from gym import spaces
import jax
from functools import partial


class VecWrapper(VectorEnv):

    def __init__(self, vec_env):
        self.vec_env = vec_env
        super().__init__(
            self.vec_env.n_envs,
            self.vec_env.single_observation_space,
            self.vec_env.single_action_space
        )

    def reset(self):
        return self.vec_env.reset()

    def step_wait(self):
        return self.vec_env.step_wait()

    def step_async(self, actions):
        return self.vec_env.step_async(actions)

    def __getattr__(self, name):
        return getattr(self.vec_env, name)


class SelectionValues(VecWrapper):

    def __init__(self, vec_env):
        super().__init__(vec_env)

        self.k = (1 + sqrt(1 + 8 * self.individual_per_gen)) // 2
        if self.k * (self.k - 1) / 2 != self.individual_per_gen:
            raise ValueError(
                "Value for individual_per_gen should ",
                "be exactly obtainable from full diallel"
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
        return self.simulator._diallel_indices(best_pop)

    def step_async(self, actions):
        super().step_async(self._convert_actions(actions))
