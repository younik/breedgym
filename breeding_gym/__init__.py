from gym.envs.registration import register

register(
    id='BreedingGym',
    entry_point='breeding_gym.breeding_gym:BreedingGym',
    max_episode_steps=50,
)