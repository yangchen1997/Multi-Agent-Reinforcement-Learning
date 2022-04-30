

# Abstract

The implementation of multi-agent reinforcement learning algorithm in Pytorch, including: Grid-Wise Control, Qmix, Centralized PPO. Different learning strategies can be specified during training, and model and experimental data can be saved.

**Quick Start:** Run the **main.py** script to start training. Please specify all parameters in the **config.yaml** file (The parameters used in this project are not optimal parameters, please adjust them according to the actual requirement).

# Petting Zoo

**MPE:** Multi Particle Environments (MPE) are a set of communication oriented environment where particle agents can (sometimes) move, communicate, see each other, push each other around, and interact with fixed landmarks.

These environments are from [OpenAI’s MPE](https://github.com/openai/multiagent-particle-envs) codebase, with several minor fixes, mostly related to making the action space discrete by default, making the rewards consistent and cleaning up the observation space of certain environments.

The environment applied in this project is **Simple Spread** (I'm also considering adding other environments in future releases).

<img src="README.assets/mpe_simple_spread.gif" alt="Env image" style="zoom:67%;" />



# Requirement

Note: The following are suggested versions only, and do not mean that the program will not work with other versions.

| Name       | Version     |
| ---------- | ----------- |
| Python     | 3.6.1       |
| gym        | 0.21.0      |
| numpy      | 1.19.1      |
| PettingZoo | 1.12.0      |
| Pytorch    | 1.6.0+cu101 |



# Corresponding Papers

- [Grid-Wise Control for Multi-Agent Reinforcement Learning in Video Game AI]([proceedings.mlr.press/v97/han19a/han19a.pdf](http://proceedings.mlr.press/v97/han19a/han19a.pdf))
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)

- [The Surprising Effectiveness of PPOin Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)



# Reference

- **petting zoo:**

```
@article{terry2020pettingzoo,
  Title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
  Author = {Terry, J. K and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sulivan, Ryan and Santos, Luis and Perez, Rodrigo and Horsch, Caroline and Dieffendahl, Clemens and Williams, Niall L and Lokesh, Yashas and Sullivan, Ryan and Ravi, Praveen},
  journal={arXiv preprint arXiv:2009.14471},
  year={2020}
}
```

- **Qmix:**  [starry-sky6688/StarCraft: Implementations of IQL, QMIX, VDN, COMA, QTRAN, MAVEN, CommNet, DyMA-CL, and G2ANet on SMAC, the decentralised micromanagement scenario of StarCraft II (github.com)](https://github.com/starry-sky6688/StarCraft)
