import gymnasium as gym
import jax

from breedgym.utils.paths import DATA_PATH

env = gym.make(
    "KBestBreedGym",
    individual_per_gen=200,
    initial_population=DATA_PATH.joinpath("small_geno.txt"),
    genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
)

jax.profiler.start_trace("profile-env", create_perfetto_trace=True)

env.reset()
for _ in range(10):
    env.step(10)
env.close()

jax.profiler.stop_trace()
