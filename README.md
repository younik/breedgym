# BreedGym

## Installation

Using pip:
```bash
pip install breedgym
```

From source:
```bash
git clone https://github.com/younik/breedgym
cd breedgym
pip install -e .
```

## Quickstart

BreedGym environments implement the [Gymnasium](https://gymnasium.farama.org) API, making it easy to use it with your preferred learning library.

```python
import gymnasium as gym
import numpy as np

env = gym.make(
    "breedgym:BreedGym",
    genetic_map="path/to/genetic_map.txt",
    initial_population="path/to/geno.npy",
    num_generations=10
)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
```
To test, you can use the sample data we provide [here](https://github.com/younik/breedgym/tree/main/breedgym/data). In the case of the small sample data, we have 370 initial population members with 10k markers. 

```
Observation space: Box(False, True, (370, 10000, 2), bool)
Action space: Sequence(Tuple(Discrete(370), Discrete(370)), stack=False)
```

After initializing the environment, we can interact with it as a standard Gymnasium environment:

```python
initial_pop, info = env.reset()
tru = False
for gen_number in range(10):
    assert not tru
    act = env.action_space.sample()
    pop, rew, ter, tru, infos = env.step(np.asarray(act))
```

After 10 generations, we expect the environment to truncate, as we specified 10 generations horizon during environment initialization:

```python
assert tru
print("Reward (GEBV mean):", rew)
```

The full list of environments can be found [here](https://github.com/younik/breedgym/blob/main/breedgym/__init__.py).

## Citing

```
@inproceedings{younis2023breeding,
  title={Breeding Programs Optimization with Reinforcement Learning},
  author={Younis, Omar G. and Corinzia, Luca and Athanasiadis, Ioannis N and Krause, Andreas and Buhmann, Joachim  and Turchetta, Matteo},
  booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
  url={https://www.climatechange.ai/papers/neurips2023/93},
  year={2023}
}
```
