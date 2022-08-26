from breeding_gym import BreedingGym, BaselineAgent

dir_path = "/home/omar/PycharmProjects/breedinggym/breeding_gym"
env = BreedingGym(
    chromosomes_map=f"{dir_path}/data/map.txt", 
    population=f"{dir_path}/data/geno.txt", 
    marker_effects=f"{dir_path}/data/marker_effects.txt"
)
agent = BaselineAgent()

obs = env.reset()

obs = env.step(agent(*obs))