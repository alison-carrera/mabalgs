import numpy as np
from mab import algs, ranked_algs
import random


class BernoulliArm:

    """

        This class generates a reward value from an uniform distribution.

    """

    def __init__(self, p):
        """
            :param p: Probability to reward an arm.
        """
        self.p = p

    def draw(self):
        """
            :return: Return a reward value.
        """
        if np.random.uniform() > self.p:
            return 0
        else:
            return 1


class MonteCarloSimulator:

    """
        This class represents a Monte Carlo Simulator for MAB debug/test.
    """

    def init_arms(self, rewards_proba):
        """
            This method reads the reward probabilities array and instantiate the bernoulli arms.

            :return: Return a list of bernoulli arms.
        """
        return list(map(lambda mu: BernoulliArm(mu), rewards_proba))

    def get_algorithm(self, name, number_of_arms):
        """
            This method instantiate the algorithm class.

            :param name: Name of algorithm.
            :param number_of_arms: Number of arms.

            :return: Returns the algorithm instance.

        """
        if name == 'ths':
            alg = algs.ThompsomSampling(number_of_arms)
        elif name == 'tuned':
            alg = algs.UCBTuned(number_of_arms)
        elif name == 'ucb1':
            alg = algs.UCB1(number_of_arms)
        return alg

    def run(self, algorithm_name, rewards_proba, numbers_of_simulations, numbers_of_pull_arms):
        """
            This is the principal method. It starts the simulation. It can be slow.

            :param algorithm_name: Algorithm name to execute.

                'ths' for Thompsom Sampling;
                'tuned' for UCB-Tuned;
                'ucb1' for UCB1.

            :param rewards_proba: A dict with key representing a time of simulation and 
            an array as a value of this dict with the probability of choosing an arm.

            :return: An array with numbers of simulations, numbers of pull_arms, arm probability at time t and cumulative rewards at time t

        """

        number_of_arms = len(rewards_proba[0])

        arm_probability = np.zeros([numbers_of_pull_arms, number_of_arms])
        cumulative_reward = np.zeros(numbers_of_pull_arms)
        cumulative_total = np.zeros(numbers_of_pull_arms)

        for s in range(1, numbers_of_simulations):
            alg = self.get_algorithm(algorithm_name, number_of_arms)

            arms = self.init_arms(rewards_proba[0])

            for t in range(1, numbers_of_pull_arms):

                if t in rewards_proba:
                    arms = self.init_arms(rewards_proba[t])

                chosen_arm = alg.select()[0]
                reward = arms[chosen_arm].draw()

                arm_probability[t, chosen_arm] = arm_probability[t,
                                                                 chosen_arm] + (1 / numbers_of_simulations)

                cumulative_reward[t] = cumulative_reward[t -
                                                         1] + (reward / numbers_of_simulations)

                cumulative_total[t] = cumulative_total[t] + \
                    cumulative_reward[t]

                if reward == 1:
                    alg.reward(chosen_arm)

        return [numbers_of_simulations, numbers_of_pull_arms, arm_probability, cumulative_total]


class RankedMonteCarloSimulator:

    """
        This class represents a Monte Carlo Simulator for  Ranlek MAB debug/test.
    """

    def init_arms(self, rewards_proba):
        """
            This method reads the reward probabilities array and instantiate the bernoulli arms.

            :return: Return a list of bernoulli arms.
        """

        final_list = []

        for arms in rewards_proba:
            final_list.append(list(map(lambda mu: BernoulliArm(mu), arms)))

        return final_list

    def get_algorithm(self, name, n_arms, n_ranks, mab_algorithm_class):
        """
            This method instantiate the algorithm class.

            :param name: Name of algorithm.
            :param n_arms: Number of arms.
            :param n_ranks: Number os ranks.
            :param mab_algorithm_class: Class reference.

            :return: Returns the algorithm instance.

        """

        if name == 'rba':
            alg = ranked_algs.RBA(n_arms, n_ranks, mab_algorithm_class)
        if name == 'rbam':
            alg = ranked_algs.RBAM(n_arms, n_ranks, mab_algorithm_class)
        return alg

    def get_rewards(self, arms, chosen_arms):
        """

          This method presents a reward calculation way to ranked mabs.

          :param arms: Arms probability in each rank.
          :param chosen_arms: The selected arms by ranked algorithm.

          :return: The rewarded arm and the value of the reward.

        """

        rewards = []
        velocity_factor = []

        for i in range(len(chosen_arms)):
            rewards.append(arms[i][chosen_arms[i]].draw())
            velocity_factor.append(arms[-1][i].draw())

        rewards = np.asarray(rewards)
        velocity_factor = np.asarray(velocity_factor)
        final_rewards = np.logical_and(rewards, velocity_factor).astype(int)

        if final_rewards.sum() > 1:
            non_zero_index = np.flatnonzero(final_rewards)
            element = random.choice(list(non_zero_index))
            return chosen_arms[element], 1
        elif final_rewards.sum() == 1:
            return chosen_arms[final_rewards.tolist().index(1)], 1
        else:
            return -1, 0

    def run(self, algorithm_name, mab_class, rewards_proba, numbers_of_simulations, numbers_of_pull_arms):
        """
            This is the principal method. It starts the simulation. It can be slow.

            :param algorithm_name: Algorithm name to execute.

                'rba' for RBA algorithm;
                'rbam' for RBAM algorithm.

            :param rewards_proba: A dict with key representing a time of simulation and 
            an array of array as a value of this dict with the probability of choosing an arm for every rank.
            The last array needs to be the weights of the ranks.

            :return: An array with numbers of simulations, numbers of pull_arms, arm probability at time t and cumulative rewards at time t

        """

        number_of_ranks = len(rewards_proba[0]) - 1
        number_of_arms = len(rewards_proba[0][0])

        arm_probability = np.zeros(
            [number_of_ranks, numbers_of_pull_arms, number_of_arms])
        cumulative_reward = np.zeros(numbers_of_pull_arms)
        cumulative_total = np.zeros(numbers_of_pull_arms)

        for s in range(1, numbers_of_simulations):
            alg = self.get_algorithm(
                algorithm_name, number_of_arms, number_of_ranks, mab_class)

            arms = self.init_arms(rewards_proba[0])

            for t in range(1, numbers_of_pull_arms):

                if t in rewards_proba:
                    arms = self.init_arms(rewards_proba[t])

                chosen_arms = alg.select()

                selected_arm, reward = self.get_rewards(arms, chosen_arms)

                for i in range(len(chosen_arms)):
                    arm_probability[i, t, chosen_arms[i]] = arm_probability[i,
                                                                            t, chosen_arms[i]] + (1 / numbers_of_simulations)

                cumulative_reward[t] = cumulative_reward[t -
                                                         1] + (reward / numbers_of_simulations)

                cumulative_total[t] = cumulative_total[t] + \
                    cumulative_reward[t]

                if selected_arm != -1:
                    alg.reward(chosen_arms, selected_arm)

        return [numbers_of_simulations, numbers_of_pull_arms, arm_probability, cumulative_total]
