from gymnasium.envs.registration import register

register(
    id='BreedingGym',
    entry_point='breeding_gym.breeding_gym:BreedingGym',
)

register(
    id='SimplifiedBreedingGym',
    entry_point='breeding_gym.wrappers:SimplifiedBreedingGym',
)

register(
    id='KBestBreedingGym',
    entry_point='breeding_gym.wrappers:KBestBreedingGym',
)
