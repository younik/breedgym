from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union
from typing_extensions import override

import jax
import numpy as np
from chromax import Simulator
from chromax.typing import Parents, Population
from gymnasium import spaces
from gymnasium.experimental.vector import AsyncVectorEnv, VectorEnv
from gymnasium.vector.utils.spaces import batch_space
from jax._src.lib import xla_client as xc
from jaxtyping import Array, Bool, Float, Int

from breedgym.utils.paths import DATA_PATH


GENOME_FILE = DATA_PATH.joinpath("small_geno.npy")


@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(None, None, 0))
def _random_selection(
    germplasm: Population["n"], length: int, key: jax.random.PRNGKeyArray
) -> Population["length"]:
    return jax.random.choice(key, germplasm, shape=(length,), replace=False)


class VecBreedGym(VectorEnv):
    def __init__(
        self,
        num_envs: int = 1,
        initial_population: Union[str, Path, Population["n"]] = GENOME_FILE,
        individual_per_gen: Optional[int] = None,
        num_generations: int = 10,
        autoreset: bool = True,
        reward_shaping: bool = False,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.num_generations = num_generations
        self.autoreset = autoreset
        self.reward_shaping = reward_shaping
        self.simulator = Simulator(**kwargs)
        self.device = self.simulator.device

        germplasm = initial_population
        if isinstance(initial_population, (str, Path)):
            germplasm = self.simulator.load_population(initial_population)

        self.germplasm = jax.device_put(germplasm, device=self.device)
        if individual_per_gen is None:
            individual_per_gen = len(self.germplasm)
        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_envs, self.individual_per_gen, self.germplasm.shape[1], 2),
            dtype=np.int8,
        )
        self.action_space = spaces.Box(
            low=0,
            high=self.individual_per_gen,
            shape=(self.num_envs, self.individual_per_gen, 2),
            dtype=np.int32,
        )

        self.populations = None
        self.step_idx = None
        self.reset_infos = {}
        self.random_key = None

    @partial(jax.vmap, in_axes=(None, 0))
    def cross(self, parents: Parents["n"]) -> Population["n"]:
        return self.simulator.cross(parents)

    def step(
        self, actions: Int[Array, "envs n 2"]
    ) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict,
    ]:
        actions = jax.device_put(actions, device=self.device)
        arange_envs = np.arange(self.num_envs)[:, None, None]
        parents = self.populations[arange_envs, actions]
        self.populations = self.cross(parents)
        self.step_idx += 1

        infos = self.get_info()
        done = self.step_idx == self.num_generations
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

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Population["envs n"], dict]:
        self.step_idx = 0
        if seed is not None:
            self.simulator.set_seed(seed)
            self.random_key = jax.random.PRNGKey(seed)
        elif self.random_key is None:
            seed = np.random.randint(2**32)
            self.random_key = jax.random.PRNGKey(seed)

        keys = jax.random.split(self.random_key, num=self.num_envs + 1)
        self.random_key = keys[0]

        if options is not None and "individual_per_gen" in options.keys():
            self.individual_per_gen = options["individual_per_gen"]

        self.populations = _random_selection(
            self.germplasm, self.individual_per_gen, keys[1:]
        )
        self.reset_infos = self.get_info()
        return self.populations, self.reset_infos

    def get_info(self) -> dict:
        GEBVs = self.simulator.GEBV_model(self.populations)
        return {"GEBV": GEBVs}

    def set_attr(self, name, values):
        return setattr(self, name, values)


class _VecBreedGym(VecBreedGym):
    def step(
        self, action
    ) -> Tuple[Population["envs n"], Float[Array, "envs"], bool, bool, dict]:
        obs, rews, ter, tru, infos = super().step(action)
        assert np.all(ter == ter[0])
        assert np.all(tru == tru[0])
        return obs, rews, ter[0], tru[0], infos


class DistributedBreedGym(AsyncVectorEnv):
    def __init__(
        self,
        envs_per_device: int,
        initial_population: Union[str, Path, Population["n"]],
        devices: Optional[List[xc.Device]] = None,
        **kwargs,
    ):
        if devices is None:
            devices = jax.local_devices()
        self.devices = devices
        self.envs_per_device = envs_per_device

        dummy_env = VecBreedGym(
            1, device=None, initial_population=initial_population, **kwargs
        )
        single_observation_space = dummy_env.single_observation_space
        single_action_space = dummy_env.single_action_space
        germplasm = dummy_env.germplasm
        del dummy_env

        device_ids = [device.id for device in self.devices]
        env_fns = [
            lambda: _VecBreedGym(
                envs_per_device,
                initial_population=germplasm,
                device=device_id,
                autoreset=False,
                **kwargs,
            )
            for device_id in device_ids
        ]

        super().__init__(env_fns, context="spawn")
        self.num_envs = self.envs_per_device * len(self.devices)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._actions = None

    def reset(self, *args, **kwargs) -> Tuple[Population["envs n"], dict]:
        obs, info = super().reset(*args, **kwargs)
        return obs.reshape(-1, *obs.shape[2:]), info

    def step_async(self, actions: Int[Array, "envs n 2"]):
        super().step_async(
            actions.reshape(len(self.devices), self.envs_per_device, *actions.shape[1:])
        )

    def step_wait(
        self, *args, **kwargs
    ) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict,
    ]:
        obs, rews, ter, tru, infos = super().step_wait(*args, **kwargs)
        obs = obs.reshape(-1, *obs.shape[2:])

        assert np.all(ter == ter[0])
        terminated = np.full((self.num_envs,), ter[0])
        assert np.all(tru == tru[0])
        truncated = np.full((self.num_envs,), tru[0])

        return obs, rews.flatten(), terminated, truncated, infos

    @override
    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        for k in info.keys():
            if k not in infos:
                type_ = type(info[k][0])
                info_array, array_mask = self._init_info_arrays(type_)
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            start_idx = env_num * self.envs_per_device
            for offset in range(0, self.envs_per_device):
                info_array[start_idx + offset] = info[k][offset]
                array_mask[start_idx + offset] = True

            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos
