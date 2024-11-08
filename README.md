# CSP: Coordination Scheme Probing for Generalizable Multi-Agent Reinforcement Learning
This is the implementation of the paper "Coordination Scheme Probing for Generalizable Multi-Agent Reinforcement Learning", based on oxwhirl's [PyMARL](https://github.com/oxwhirl/pymarl.git) for multi-agent reinforcement learning.

Note: the experiments of CSP are conducted in SC2.4.6.2.69232, which is same as the [SMAC run data release](https://github.com/oxwhirl/smac/releases/tag/v1). The results are not always comparable with results run in SC2.4.10.

## Installation instructions

The codebase is only tested on Ubuntu 18.04 and 20.04 LTS. Anaconda should be installed in advance before setting up environments.

Set up with our bash file:

```shell
sh install_env.sh
```

You can modify this file to skip the installation of StarCraft II if it is already installed.

## Run experiments

All experiments can be run by the following command:

```shell
python src/main.py --config=[config_file]
```

The [config_file] should be the name of a `.yaml` file placed in the `src/config/exps/` folder. We have included some samples in it of different training stages in each environment.

For Stage 1:

- Choose config file `stage1_xx`

For Stage 2: 

- Choose config file `stage2_xx`
- Modify **population_directories** to where you save teammate policies trained in Stage 1

For Stage 3: 

- Choose config file `stage3_xx`
- Modify **population_directories** to where you save teammate policies trained in Stage 1
- Modify **explore_load_path** to where you save team probing module trained in Stage 2
