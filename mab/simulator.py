import numpy as np
from mab import algs


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
        return list(map(lambda mu: BernoulliArm(rewards_proba), rewards_proba))
    
    def get_algorithm(self, name, number_of_arms):
        """
            This method instantiate the algorithm class.

            :param name: Name of algorithm.
            :param number_of_arms: Class reference.

            :return: Returns the algorithm instance.

        """
        if name == 'ths':
            alg = algs.ThompsomSampling(number_of_arms)
        elif name == 'tuned':
            alg = algs.UCBTuned(number_of_arms)
        elif name == 'ucb1':
            alg = algs.UCB1(number_of_arms)
        return alg

    def run(self, algorithm_name, rewards_proba, number_of_arms, numbers_of_simulations, numbers_of_pull_arms):
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
        
        arm_probability = np.zeros([numbers_of_pull_arms, number_of_arms])
        cumulative_reward = np.zeros(numbers_of_pull_arms)
        cumulative_total = np.zeros(numbers_of_pull_arms)        
        
        for s in range(1, numbers_of_simulations):
            alg = self.get_algorithm(algorithm_name, number_of_arms)

            arms = self.init_arms(rewards_proba[0])
            
            for t in range(1, numbers_of_pull_arms):
                
                if t in rewards_proba:
                  arms = self.init_arms(rewards_proba[t])

                chosen_arm = alg.select()
                reward = arms[chosen_arm].draw()

                arm_probability[t, chosen_arm] = arm_probability[t, chosen_arm] + (1 / numbers_of_simulations)
                
                cumulative_reward[t] = cumulative_reward[t - 1] + (reward / numbers_of_simulations)
                
                cumulative_total[t] = cumulative_total[t] + cumulative_reward[t]
                
                if algorithm_name == 'ths' and reward == 0:
                  alg.penalty(chosen_arm)
                
                if reward == 1:                
                  alg.reward(chosen_arm)

        
        return [numbers_of_simulations, numbers_of_pull_arms, arm_probability, cumulative_total]           