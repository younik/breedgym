from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Tuple
from typing_extensions import override
from gym import spaces
from gym.vector.vector_env import VectorEnv
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.utils.spaces import batch_space
import jax
from jax._src.lib import xla_client as xc
from jaxtyping import Array, Bool, Float, Int
import numpy as np
from breeding_gym.simulator import BreedingSimulator
from breeding_gym.simulator.typing import Population, Parents
from breeding_gym.utils.paths import DATA_PATH


GENOME_FILE = DATA_PATH.joinpath("geno.txt")


@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(None, None, 0))
def _random_selection(
    germplasm: Population["n"],
    length: int,
    key: jax.random.PRNGKeyArray
) -> Population["length"]:
    return jax.random.choice(
        key,
        germplasm,
        shape=(length,),
        replace=False
    )


class VecBreedingGym(VectorEnv):

    MAX_EPISODE_STEPS = 10

    def __init__(
        self,
        n_envs: int,
        initial_population: Union[str, Path, Population["n"]] = GENOME_FILE,
        individual_per_gen: Optional[int] = None,
        autoreset: bool = True,
        **kwargs
    ):
        self.n_envs = n_envs
        self.autoreset = autoreset
        self.simulator = BreedingSimulator(**kwargs)
        self.device = self.simulator.device

        germplasm = initial_population
        if isinstance(initial_population, (str, Path)):
            germplasm = self.simulator.load_population(initial_population)

        self.germplasm = jax.device_put(germplasm, device=self.device)
        if individual_per_gen is None:
            individual_per_gen = len(self.germplasm)
        self.individual_per_gen = individual_per_gen

        observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.individual_per_gen, self.germplasm.shape[1], 2),
            dtype=np.int8
        )
        action_space = spaces.Sequence(
            spaces.Tuple((
                spaces.Discrete(self.individual_per_gen),
                spaces.Discrete(self.individual_per_gen)
            ))
        )
        super().__init__(n_envs, observation_space, action_space)

        self.populations = None
        self.step_idx = None
        self.reset_infos = {}
        self._actions = None
        self.random_key = None

    @partial(jax.vmap, in_axes=(None, 0))
    def _cross(self, parents: Parents["n"]) -> Population["n"]:
        return self.simulator.cross(parents)

    def step_async(self, actions: Int[Array, "envs n 2"]):
        self._actions = jax.device_put(actions, device=self.device)

    def step_wait(self) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict
    ]:
        arange_envs = np.arange(self.n_envs)[:, None, None]
        parents = self.populations[arange_envs, self._actions]
        self.populations = self._cross(parents)
        self.step_idx += 1

        infos = self._get_info()
        done = self.step_idx == self.MAX_EPISODE_STEPS
        if done:
            rews = np.mean(infos["GEBV"], axis=(1, 2))
            rews = np.asarray(rews)
            if self.autoreset:
                self.reset()
        else:
            rews = np.zeros(self.n_envs)

        terminated = np.full(self.n_envs, False)
        truncated = np.full(self.n_envs, done)
        return self.populations, rews, terminated, truncated, infos

    def reset_async(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        self.step_idx = 0
        if seed is not None:
            self.simulator.set_seed(seed)
            self.random_key = jax.random.PRNGKey(seed)
        elif self.random_key is None:
            seed = np.random.randint(2**32)
            self.random_key = jax.random.PRNGKey(seed)

        keys = jax.random.split(self.random_key, num=self.n_envs + 1)
        self.random_key = keys[0]

        if options is not None and "individual_per_gen" in options.keys():
            self.individual_per_gen = options["individual_per_gen"]

        self.populations = _random_selection(
            self.germplasm,
            self.individual_per_gen,
            keys[1:]
        )
        self.reset_infos = self._get_info()

    def reset_wait(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Population["envs n"], dict]:
        return self.populations, self.reset_infos

    def _get_info(self) -> dict:
        GEBVs = self.simulator.GEBV_model(self.populations)
        return {"GEBV": GEBVs}


class _VecBreedingGym(VecBreedingGym):

    def step_wait(self) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        bool,
        bool,
        dict
    ]:
        obs, rews, ter, tru, infos = super().step_wait()
        assert np.all(ter == ter[0])
        assert np.all(tru == tru[0])
        return obs, rews, ter[0], tru[0], infos


class DistributedBreedingGym(AsyncVectorEnv):

    def __init__(
        self,
        envs_per_device: int,
        initial_population: Union[str, Path, Population["n"]],
        devices: Optional[List[xc.Device]] = None,
        **kwargs
    ):
        if devices is None:
            devices = jax.local_devices()
        self.devices = devices
        self.envs_per_device = envs_per_device

        dummy_env = VecBreedingGym(
            1,
            device=None,
            initial_population=initial_population,
            **kwargs
        )
        single_observation_space = dummy_env.single_observation_space
        single_action_space = dummy_env.single_action_space
        germplasm = dummy_env.germplasm
        del dummy_env

        device_ids = [device.id for device in self.devices]
        env_fns = [
            lambda: _VecBreedingGym(
                envs_per_device,
                device_id,
                autoreset=False,
                initial_population=germplasm,
                **kwargs
            )
            for device_id in device_ids
        ]

        super().__init__(env_fns, context="spawn")
        self.num_envs = self.envs_per_device * len(self.devices)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(
            self.single_observation_space,
            self.num_envs
        )
        self.action_space = batch_space(
            self.single_action_space,
            self.num_envs
        )
        self._actions = None

    def reset(self, *args, **kwargs) -> Tuple[Population["envs n"], dict]:
        obs, info = super().reset(*args, **kwargs)
        return obs.reshape(-1, *obs.shape[2:]), info

    def step_async(self, actions: Int[Array, "envs n 2"]):
        super().step_async(
            actions.reshape(
                len(self.devices),
                self.envs_per_device,
                *actions.shape[1:]
            )
        )

    def step_wait(self, *args, **kwargs) -> Tuple[
        Population["envs n"],
        Float[Array, "envs"],
        Bool[Array, "envs"],
        Bool[Array, "envs"],
        dict
    ]:
        obs, rews, ter, tru, infos = super().step_wait(*args, **kwargs)
        obs = obs.reshape(-1, *obs.shape[2:])

        assert np.all(ter == ter[0])
        terminated = np.full((self.num_envs, ), ter[0])
        assert np.all(tru == tru[0])
        truncated = np.full((self.num_envs, ), tru[0])

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
