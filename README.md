# A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values

Official implementation of Shapley Values for Explaining Reinforcement Learning (SVERL) | [2025 journal version]()

Daniel Beechey, Thomas M. S. Smith, Özgür Şimşek

## Abstract

Reinforcement learning agents can achieve superhuman performance, but their decisions are often difficult to interpret. This lack of transparency limits deployment, especially in high-stakes settings where human trust and accountability are essential. In this work, we develop a theoretical framework for explaining reinforcement learning through the influence of features---the values that describe an agent’s state. We identify three core elements of agent-environment interaction that benefit from explanation: an agent’s behaviour (what the agent does), performance (what it achieves), and value estimation (what it expects to achieve). To explain each, we treat features as players cooperating to produce these quantities, and apply Shapley values---a principled method from cooperative game theory---to attribute influence. This yields a family of mathematically grounded explanations with clear semantics and theoretical guarantees. We use illustrative examples to show how these explanations align with human intuition and reveal novel insights. Our framework unifies and extends prior work, clarifies the assumptions behind existing approaches, and offers a foundation for more interpretable and trustworthy reinforcement learning.

## Overview

This github enables the replication of the experiments in the 2025 paper: A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values.

If you need help to use SVERL, please open an issue or contact djeb20@bath.ac.uk.

## Requirements

## Citation

If you use find this code useful for your research, please consider citing our works:

```
@inproceedings{beechey2023explaining,
  title={Explaining reinforcement learning with shapley values},
  author={Beechey, Daniel and Smith, Thomas MS and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r},
  booktitle={International Conference on Machine Learning},
  pages={2003--2014},
  year={2023},
  organization={PMLR}
}
```
