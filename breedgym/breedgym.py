from math import ceil, floor, sqrt
from pathlib import Path
from typing import Optional, Tuple, Union

import gymnasium as gym
import jax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chromax import Simulator
from chromax.typing import Population
from gymnasium import spaces
from jaxtyping import Array, Float, Int

from breedgym.utils.paths import DATA_PATH
from breedgym.utils.plot_utils import NEURIPS_FONT_FAMILY, set_up_plt


GENOME_FILE = DATA_PATH.joinpath("small_geno.npy")


class BreedGym(gym.Env):

    metadata = {"render_modes": ["matplotlib"], "render_fps": 1}

    def __init__(
        self,
        initial_population: Union[Path, Population["n"]] = GENOME_FILE,
        num_generations: int = 10,
        reward_shaping: bool = False,
        render_mode: Optional[str] = None,
        render_kwargs: dict = {},
        **kwargs
    ):
        self.simulator = Simulator(**kwargs)
        if isinstance(initial_population, (str, Path)):
            germplasm = self.simulator.load_population(initial_population)
        else:
            germplasm = initial_population
        self.device = self.simulator.device
        self.germplasm = jax.device_put(germplasm, device=self.device)

        self.num_generations = num_generations
        self.reward_shaping = reward_shaping

        self.observation_space = spaces.Box(
            0, 1, shape=self.germplasm.shape, dtype=np.bool_
        )
        self.action_space = spaces.Sequence(
            spaces.Tuple(
                (
                    spaces.Discrete(len(self.germplasm)),
                    spaces.Discrete(len(self.germplasm)),
                )
            )
        )

        self.population = None
        self._GEBV, self._GEBV_cache = None, False
        self._corrcoef, self._corrcoef_cache = None, False

        self.step_idx = None
        self.episode_idx = -1
        self.render_mode = render_mode
        if self.render_mode is not None:
            self.render_kwargs = render_kwargs
            self.render_kwargs.setdefault("colors", ["b", "g", "r", "c", "m"])
            self.render_kwargs.setdefault("offset", 0)
            self.render_kwargs.setdefault("traits", self.simulator.trait_names)
            self.render_kwargs.setdefault(
                "other_features",
                [
                    lambda: self.corrcoef,
                ],
            )
            self.render_kwargs.setdefault(
                "feature_names",
                [
                    "corrcoef",
                ],
            )
            self.render_kwargs.setdefault("episode_names", "Episode {:d}")
            self.render_kwargs.setdefault("font", NEURIPS_FONT_FAMILY)
            self.axs = self._make_axs()

    def _make_axs(self):
        if "axs" in self.render_kwargs.keys():
            return self.render_kwargs["axs"]
        else:
            n_figs = len(self.render_kwargs["traits"]) + len(
                self.render_kwargs["other_features"]
            )
            nrows = floor(sqrt(n_figs))
            ncols = ceil(n_figs / nrows)
            axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))[1]
            return axs.flatten()

    def _update_spaces(self):
        self.observation_space = spaces.Box(
            0, 1, shape=self.population.shape, dtype=np.bool_
        )
        self.action_space = spaces.Sequence(
            spaces.Tuple(
                (
                    spaces.Discrete(len(self.population)),
                    spaces.Discrete(len(self.population)),
                )
            )
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Population["n"], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.simulator.set_seed(seed=seed)

        self.step_idx = 0
        self.episode_idx += 1
        if options is not None and "n_individuals" in options.keys():
            selected_indices = self.np_random.choice(
                len(self.germplasm), options["n_individuals"], replace=False
            )
            self.population = self.germplasm[selected_indices]
        else:
            self.population = self.germplasm

        self._update_spaces()
        info = self._get_info()
        if self.render_mode is not None:
            self._render_step(info)

        return self.population, info

    def step(
        self, action: Int[Array, "n 2"]
    ) -> Tuple[Population["n"], float, bool, bool, dict]:
        """Action is an array of shape n x 2, where n is the number of crosses.
        Each row contains a couple of parent indices.
        """
        parents = self.population[action]  # n x 2 x markers x 2
        self.population = self.simulator.cross(parents)
        self.step_idx += 1
        self._update_spaces()

        info = self._get_info()
        if self.render_mode is not None:
            self._render_step(info)

        truncated = self.step_idx == self.num_generations
        if self.reward_shaping or truncated:
            reward = np.mean(self.GEBV.to_numpy())
        else:
            reward = 0

        return self.population, reward, False, truncated, info

    def _render_step(self, info: dict):
        def boxplot(axs, values):
            bp = axs.boxplot(
                values,
                positions=[self.step_idx + self.episode_idx / 6],
                flierprops={"markersize": 2},
            )
            color = self.render_kwargs["colors"][self.episode_idx]
            plt.setp(bp.values(), color=color)

        for idx, trait in enumerate(self.render_kwargs["traits"]):
            boxplot(self.axs[idx], info["GEBV"][trait])

        for feature in self.render_kwargs["other_features"]:
            idx += 1
            boxplot(self.axs[idx], feature())

    def render(self, file_name: Optional[Union[str, Path]] = None):
        if self.render_mode is not None:
            set_up_plt(self.render_kwargs["font"], use_tex=False)

            xticks = np.arange(self.step_idx + 1)

            titles = self.render_kwargs["traits"] + self.render_kwargs["feature_names"]
            for ax, title in zip(self.axs, titles):
                ax.set_xticks(xticks + self.episode_idx / 12, xticks)
                ax.set_title(title)
                ax.grid(axis="y")
                ax.set_xlabel("Generations [Years]")

            len_episodes = self.episode_idx + 1
            episode_names = self.render_kwargs["episode_names"]
            if not isinstance(episode_names, list):
                episode_names = [episode_names] * len_episodes

            def rectangle(ep_idx):
                return mpatches.Rectangle(
                    (0, 1),
                    color=self.render_kwargs["colors"][ep_idx],
                    label=episode_names[ep_idx].format(ep_idx),
                    width=0.1,
                    height=0.1,
                    fill=False,
                )

            patches = [rectangle(idx) for idx in range(len_episodes)]
            plt.figlegend(handles=patches, loc="upper right")

            plt.tight_layout()
            if file_name is not None:
                plt.savefig(file_name, bbox_inches="tight")
            plt.show()

    def _get_info(self):
        return {"GEBV": self.GEBV}

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, new_pop: Population["n"]):
        self._population = new_pop
        self._GEBV_cache = False
        self._corrcoef_cache = False

    @property
    def GEBV(self) -> pd.DataFrame:
        """
        Returns the GEBV for each traits of each individuals.
        If the population is composed by n individual,
        the output will be n x t, where t is the number of traits.
        """
        if not self._GEBV_cache:  # WARN: not multithreading safe
            self._GEBV = self.simulator.GEBV(self.population)
            self._GEBV_cache = True
        return self._GEBV

    @property
    def corrcoef(self) -> Float[Array, "n"]:
        if not self._corrcoef_cache:  # WARN: not multithreading safe
            self._corrcoef = self.simulator.corrcoef(self.population)
            self._corrcoef_cache = True
        return self._corrcoef
