# A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values

Official implementation of Shapley Values for Explaining Reinforcement Learning (SVERL) | [2025 extended version](https://arxiv.org/pdf/2505.07797)

Daniel Beechey, Thomas M. S. Smith, Özgür Şimşek

## Abstract

Reinforcement learning agents can achieve superhuman performance, but their decisions are often difficult to interpret. This lack of transparency limits deployment, especially in safety-critical settings where human trust and accountability are essential. In this work, we develop a theoretical framework for explaining reinforcement learning through the influence of state features---what the agent observes in its environment. We identify three core elements of the agent-environment interaction that benefit from explanation: behaviour (what the agent does), performance (what the agent achieves), and value estimation (what the agent expects to achieve). We treat state features as players cooperating to produce each element and apply Shapley values, a principled method from cooperative game theory, to identify the influence of each feature. This approach yields a family of mathematically grounded explanations with clear semantics and theoretical guarantees. We use illustrative examples to show how these explanations align with human intuition and reveal novel insights. Our framework unifies and extends prior work, making explicit the assumptions behind existing approaches, and offers a principled foundation for more interpretable and trustworthy reinforcement learning.

## Overview

This github enables the replication of the experiments in the 2025 paper: A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values.

If you need help to use SVERL, please open an issue or contact djeb20 AT bath DOT ac DOT uk.

## Requirements

## Citation

If you use find this code useful for your research, please consider citing our works:

```
@article{beechey2025theoretical,
  title={A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values},
  author={Beechey, Daniel and Smith, Thomas and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r},
  journal={arXiv preprint arXiv:2505.07797},
  year={2025}
}
```
