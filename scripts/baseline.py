import gym
from breeding_gym.utils.paths import DATA_PATH
import numpy as np


if __name__ == '__main__':
    individual_per_gen = 200
    trials = 100
    #actions = [10] * 10 

    env = gym.make("breeding_gym:SimplifiedBreedingGym",
                   initial_population=DATA_PATH.joinpath("small_const_chr_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_const_chr_genetic_map.txt"),
                   individual_per_gen=individual_per_gen,
                   num_generations=10,
                   reward_shaping=False
                   )

    buffer_gg = np.zeros((trials))
    for trial_idx in range(trials):
        print(trial_idx, flush=True)
        env.reset()

        tru = False
        ter = False
        #acts = iter(actions)
        while not (tru or ter):
            _, rew, ter, tru, _ = env.step({"n_bests": 20, "n_crosses": 10})
            buffer_gg[trial_idx] += rew

    print(buffer_gg.std())
    print(buffer_gg.mean())
    env.close()
