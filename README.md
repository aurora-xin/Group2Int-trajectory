# Learning Group Interactions and Semantic Intentions for Multi-Object Trajectory Prediction

This is the official implementation of our paper [Learning Group Interactions and Semantic Intentions for Multi-Object Trajectory Prediction](https://arxiv.org/abs/2412.15673).

## Introduction

Effective modeling of group interactions and dynamic semantic intentions is crucial for forecasting behaviors like trajectories or movements. In complex scenarios like sports, agents' trajectories are influenced by group interactions and intentions, including team strategies and opponent actions. To this end, we propose a novel diffusion-based trajectory prediction framework that integrates group-level interactions into a conditional diffusion model, enabling the generation of diverse trajectories aligned with specific group activity. To capture dynamic semantic intentions, we frame group interaction prediction as a cooperative game, using Banzhaf interaction to model cooperation trends. We then fuse semantic intentions with enhanced agent embeddings, which are refined through both global and local aggregation. Furthermore, we expand the NBA SportVU dataset by adding human annotations of team-level tactics for trajectory and tactic prediction tasks. Extensive experiments on three widely-adopted datasets demonstrate that our model outperforms state-of-the-art methods.


## Table of Contents

1. [News](#news)
2. [Data Preparation](#data-prep)
3. [Train and Eval](#train-and-eval)
4. [Citation](#citation)

## News <a name="news"></a>
**[20 Dec, 2024]** We have released the Arxiv version of the paper. Code/Models are coming soon. Please stay tuned! 

## Data Preparation <a name="data-prep"></a>

a. We extracted and processed data from the [NBA SportsVU dataset](https://github.com/linouk23/NBA-Player-Movements
). 

b. We extended th human annotations of team-level tactics for trajectory and tactic prediction tasks.

**TODO:** The extended dataset will be available once the paper is accepted. Stay tuned!


## Train and Eval <a name="train-and-eval"></a>
**TODO:** Code/Models will be provided here.


## Citation <a name="citation"></a>
If you find this project useful in your research, please consider citing:

```
@article{grouplearning2024,
  title={Learning Group Interactions and Semantic Intentions for Multi-Object Trajectory Prediction},
  author={Qi, Mengshi and Yang, Yuxin and Ma, Huadong},
  journal={arXiv preprint arXiv:2412.15673},
  year={2024}
}
```
