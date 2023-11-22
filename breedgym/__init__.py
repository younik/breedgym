from gymnasium.envs.registration import register

register(
    id="BreedGym",
    entry_point="breedgym.breedgym:BreedGym",
)

register(
    id="SimplifiedBreedGym",
    entry_point="breedgym.wrappers:SimplifiedBreedGym",
)

register(
    id="KBestBreedGym",
    entry_point="breedgym.wrappers:KBestBreedGym",
)
