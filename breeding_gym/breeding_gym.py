from math import sqrt, ceil, floor
import gym
from gym import spaces
import numpy as np
from breeding_gym.simulator.simulator import BreedingSimulator
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.utils.plot_utils import set_up_plt, NEURIPS_FONT_FAMILY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class BreedingGym(gym.Env):

    metadata = {"render_modes": ["matplotlib"], "render_fps": 1}
    MAX_INDIVIDUALS = np.iinfo(np.int64).max

    def __init__(
        self,
        initial_population=DATA_PATH.joinpath("geno.txt"),
        render_mode=None,
        render_kwargs={},
        **kwargs
    ):
        self.observation_space = spaces.Sequence(
            spaces.Box(0, 1, shape=(19864,))
        )
        self.action_space = spaces.Sequence(
            spaces.Tuple((
                spaces.Discrete(self.MAX_INDIVIDUALS),
                spaces.Discrete(self.MAX_INDIVIDUALS)
            ))
        )

        self.simulator = BreedingSimulator(**kwargs)
        self.germplasm = np.loadtxt(initial_population, dtype='bool')
        self.germplasm = self.germplasm.reshape(self.germplasm.shape[0], -1, 2)

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
            self.render_kwargs.setdefault("corrcoef", True)
            self.render_kwargs.setdefault("traits", self.simulator.trait_names)
            self.render_kwargs.setdefault("episode_names", "Episode {:d}")
            self.render_kwargs.setdefault("font", NEURIPS_FONT_FAMILY)
            self.axs = self._make_axs()

    def _make_axs(self):
        if "axs" in self.render_kwargs.keys():
            return self.render_kwargs["axs"]
        else:
            corrcoef = int(self.render_kwargs["corrcoef"])
            n_figs = len(self.render_kwargs["traits"]) + corrcoef
            nrows = floor(sqrt(n_figs))
            ncols = ceil(n_figs / nrows)
            axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))[1]
            return axs.flatten()

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self.step_idx = 0
        self.episode_idx += 1
        if options is not None and "n_individuals" in options.keys():
            selected_indices = np.random.choice(
                len(self.germplasm),
                options["n_individuals"],
                replace=False
            )
            self.population = self.germplasm[selected_indices]
        else:
            self.population = self.germplasm

        info = self._get_info()
        if self.render_mode is not None:
            self._render_step(info)

        if return_info:
            return self.population, info
        else:
            return self.population

    def step(self, action):
        """Action is an array of shape n x 2, where n is the number of crosses.
           Each row contains a couple of parent indices.
        """
        parents = self.population[action]  # n x 2 x markers x 2
        self.population = self.simulator.cross(parents)
        self.step_idx += 1

        info = self._get_info()
        if self.render_mode is not None:
            self._render_step(info)

        reward = np.mean(info["GEBV"]["Yield"])
        return self.population, reward, False, False, info

    def _render_step(self, info):
        def boxplot(axs, values):
            bp = axs.boxplot(
                values,
                positions=[self.step_idx + self.episode_idx / 6],
                flierprops={'markersize': 2}
            )
            color = self.render_kwargs["colors"][self.episode_idx]
            plt.setp(bp.values(), color=color)

        for idx, trait in enumerate(self.render_kwargs["traits"]):
            boxplot(self.axs[idx], info["GEBV"][trait])

        if self.render_kwargs["corrcoef"]:
            boxplot(self.axs[idx+1], self.corrcoef)

    def render(self, file_name=None):
        if self.render_mode is not None:
            set_up_plt(self.render_kwargs["font"])

            xticks = np.arange(self.step_idx + 1)

            titles = self.render_kwargs["traits"]
            if self.render_kwargs["corrcoef"]:
                titles = self.render_kwargs["traits"] + ["Corrcoef"]
            for ax, title in zip(self.axs, titles):
                ax.set_xticks(xticks + self.episode_idx / 12, xticks)
                ax.set_title(title)
                ax.grid(axis='y')
                ax.set_xlabel('Generations [Years]')

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
                    fill=False
                )
            patches = [rectangle(idx) for idx in range(len_episodes)]
            plt.figlegend(handles=patches, loc='upper right')

            plt.tight_layout()
            if file_name is not None:
                plt.savefig(file_name, bbox_inches='tight')
            plt.show()

    def _get_info(self):
        return {"GEBV": self.GEBV}

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, new_pop):
        self._population = new_pop
        self._GEBV_cache = False
        self._corrcoef_cache = False

    @property
    def GEBV(self):
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
    def corrcoef(self):
        if not self._corrcoef_cache:  # WARN: not multithreading safe
            self._corrcoef = self.simulator.corrcoef(self.population)
            self._corrcoef_cache = True
        return self._corrcoef


def default_f_index(GEBV):
    GEBV_copy = np.copy(GEBV)
    GEBV_copy[:, 2] = np.abs(GEBV_copy[:, 2])

    weights = np.array([2, -1, 1, 1, 1])
    return np.dot(GEBV_copy, weights)


class SimplifiedBreedingGym(gym.Wrapper):

    metadata = {"render_modes": ["matplotlib"], "render_fps": 1}

    def __init__(
        self,
        env=None,
        individual_per_gen=2250,
        f_index=default_f_index,
        **kwargs
    ):
        if env is None:
            env = BreedingGym(**kwargs)

        super().__init__(env)
        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Dict({
            "GEBV": spaces.Box(-15, 15, shape=(self.individual_per_gen,)),
            "corrcoef": spaces.Box(-0.5, 0.5, shape=(self.individual_per_gen,))
        })

        # max x, s.t. x * (x - 1) / 2 < individual_per_gen
        max_best = (1 + sqrt(1 + 8 * self.individual_per_gen)) // 2
        self.action_space = spaces.Discrete(int(max_best - 1), start=2)

        self.f_index = f_index

    def reset(self, seed=None, return_info=False, options=None):
        if options is None:
            options = {}
        options["n_individuals"] = self.individual_per_gen

        _, info = self.env.reset(seed, True, options)
        print(self.env.episode_idx)

        if return_info:
            return self._simplified_obs(info), info
        else:
            return self._simplified_obs(info)

    def step(self, action):
        children = action * (action - 1) / 2
        n_offspring = ceil(self.individual_per_gen / children)

        indices = self.f_index(self.GEBV)

        # retrieve the `action` best population indices
        best_pop = np.argpartition(indices, -action)[-action:]

        mesh1, mesh2 = np.meshgrid(best_pop, best_pop)
        triu_indices = np.triu_indices(action, k=1)
        mesh1 = mesh1[triu_indices]
        mesh2 = mesh2[triu_indices]
        low_level_action = np.stack([mesh1, mesh2], axis=1)
        low_level_action = np.repeat(low_level_action, n_offspring, axis=0)
        low_level_action = low_level_action[:self.individual_per_gen]

        _, _, terminated, truncated, info = self.env.step(low_level_action)
        obs = self._simplified_obs(info)
        return obs, np.mean(obs["GEBV"]), terminated, truncated, info

    def _simplified_obs(self, info):
        corrcoef = self.env.corrcoef - 0.5
        clipped_GEBV = np.clip(info["GEBV"]["Yield"], 0, 30) - 15
        return {"GEBV": clipped_GEBV.to_numpy(), "corrcoef": corrcoef}
