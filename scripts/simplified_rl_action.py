import gym
import breeding_gym  # noqa: F401
from stable_baselines3 import DQN


env = gym.make("SimplifiedBreedingGym")
model = DQN('MultiInputPolicy', env)
model.learn(total_timesteps=100)


obs = env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action)
    print(reward)
    if done:
        obs = env.reset()
