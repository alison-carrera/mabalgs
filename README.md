# Multi-Armed Bandit Algorithms
Multi-Armed Bandit (MAB) is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice.

In the problem, each machine provides a random reward from a probability distribution specific to that machine. The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls. The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines. The trade-off between exploration and exploitation is also faced in machine learning.

The main problems that the MAB help to solve is the split of the population in online experiments.


## Installing
```
pip install mabalgs
```

## Algorithms (Bandit strategies)

### UCB1 (Upper Confidence Bound)
Is an algorithm for the multi-armed bandit that achieves regret that grows only logarithmically with the number of actions taken, with no prior knowledge of the reward distribution required.

#### Get a selected arm
```python
from mab import algs

ucb_with_two_arms = algs.UCB1(2)
ucb_with_two_arms.select()
```

#### Reward an arm
```python
from mab import algs

ucb_with_two_arms = algs.UCB1(2)
my_arm = ucb_with_two_arms.select()
ucb_with_two_arms.reward(my_arm)
```
----------------

## References
- [Wikipedia MAB](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [A Survey of Online Experiment Design
with the Stochastic Multi-Armed Bandit](https://arxiv.org/pdf/1510.00757.pdf)
- [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023%2FA%3A1013689704352?LI=true)