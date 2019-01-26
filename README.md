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

# Constructor receives number of arms.
ucb_with_two_arms = algs.UCB1(2)
ucb_with_two_arms.select()
```

#### Reward an arm
```python
from mab import algs

# Constructor receives number of arms.
ucb_with_two_arms = algs.UCB1(2)
my_arm = ucb_with_two_arms.select()
ucb_with_two_arms.reward(my_arm)
```

### UCB-Tuned (Upper Confidence Bound Tuned)
A strict improvement over both UCB solutions can be made by tuning the upper-bound parameter in UCB1’s decision rule. UCB-Tuned empirically outperforms UCB1 and UCB2 in terms of frequency
of picking the best arm. Further, indicate that UCB-Tuned is “not very” sensitive to the variance of the arms. 

#### Get a selected arm
```python
from mab import algs

# Constructor receives number of arms.
ucbt_with_two_arms = algs.UCBTuned(2)
ucbt_with_two_arms.select()
```

#### Reward an arm
```python
from mab import algs

# Constructor receives number of arms.
ucbt_with_two_arms = algs.UCBTuned(2)
my_arm = ucbt_with_two_arms.select()
ucbt_with_two_arms.reward(my_arm)
```

### Thompson Sampling
Thompson Sampling is fully Bayesian: it generates a bandit configuration (i.e. a vector of expected rewards) from a posterior distribution, and then acts as if this was the true configuration (i.e. it pulls the lever with the highest expected reward).

“On the likelihood that one unknown probability exceeds another
in view of the evidence of two samples” produced the first paper on an equivalent problem to the multi-armed bandit in which a solution to the Bernoulli
distribution bandit problem now referred to as Thompson sampling is presented.

#### Get a selected arm
```python
from mab import algs

# Constructor receives number of arms.
thomp_with_two_arms = algs.ThompsomSampling(2)
thomp_with_two_arms.select()
```

#### Reward an arm
```python
from mab import algs

# Constructor receives number of arms.
thomp_with_two_arms = algs.ThompsomSampling(2)
my_arm = thomp_with_two_arms.select()
thomp_with_two_arms.reward(my_arm)
```

#### Penalty an arm

```python
# Thompsom Sampling has a penalty function. 
# It should be used in a onDestroy() event from a banner, for example. 
# The arm was selected, showed to the user, but no interation was realized until the end of the arm cycle.
thomp_with_two_arms.penalty(my_arm)
```

## Comparison of the algorithms using Monte Carlo Simulation

Monte Carlo simulation is a program for debugging / testing MAB algorithms. This evaluation was done in real time
with the aid of an execution attempt over time.

Example: We want to test a 5-arm MAB that will be used in an ad problem, and MAB must choose which of the 5 ads must
receive the most clicks from users. You can use the following probability setting ([0.9, 0.1, 0.1, 0.1, 0.1]).
The definition of each array element represents an arm and its probability of being clicked.
We can observe that Ad 0 (index 0 of array) has 90% chance of clicks while others have 10% chances of clicks.
These information can help us to analyze if the algorithm is performing well.

A simulation with the following settings was made:

```
{0: [0.9, 0.6, 0.2, 0.2, 0.1], 1000: [0.3, 0.8, 0.2, 0.2, 0.2], 4000: [0.7, 0.3, 0.2, 0.2, 0.1]}
```

The key of the dictionary tells us the time which the probabilities must be activated in the passing time.
The total time of the simulation was 5000 steps.

From this dictionary we can infer that:

- From time 0 to 1000, arm 0 is the winner.
- From time 1000 to 4000, arm 1 is the winner
- From time 4000, arm 0 is the winner.

### Results:

![UCB1](./readme-images/ucb1.png)

![UCB-Tuned](./readme-images/ucbt.png)

![Thompsom Sampling](./readme-images/ths.png)

![Cumulative Rewards](./readme-images/rewards.png)

Remembering that all these analyzes were performed in a simulation environment and the results may vary according 
to the type of information the MAB will perform on. For a more sensible choice with real world data, please perform 
an AB test between the algorithms in your scenario.

----------------

## References
- [Wikipedia MAB](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [A Survey of Online Experiment Design
with the Stochastic Multi-Armed Bandit](https://arxiv.org/pdf/1510.00757.pdf)
- [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023%2FA%3A1013689704352?LI=true)
- [Solving multiarmed bandits: A comparison of epsilon-greedy and Thompson sampling](https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50)