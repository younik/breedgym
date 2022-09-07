import gym
import breeding_gym  # noqa: F401
from stable_baselines3 import DQN


train_env = gym.make("SimplifiedBreedingGym")
model = DQN('MultiInputPolicy', train_env, buffer_size=80_000)
model.learn(total_timesteps=1000)

env = gym.make("SimplifiedBreedingGym",
               render_mode="matplotlib",
               render_kwargs={"episode_names": ["Steven's baseline", "RL"]}
               )
obs = env.reset()
for i in range(10):
    obs, reward, done, info = env.step(10)

obs = env.reset()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action, sep=", ")

env.render()
