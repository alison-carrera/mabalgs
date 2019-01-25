import numpy as np


class UCB1(object):
    def __init__(self, n_arms):
        self.number_of_selections = np.zeros(n_arms).astype(np.float)
        self.rewards = np.zeros(n_arms).astype(np.float)

    def select(self):
        arm_dont_usage = np.where(self.number_of_selections == 0)[0]
        if len(arm_dont_usage) > 0:
            self.number_of_selections[arm_dont_usage[0]] += 1
            return arm_dont_usage[0]

        average_reward = self.rewards / self.number_of_selections
        total_counts = np.sum(self.number_of_selections)

        ucb_values = self._factor_importance_each_arm(
            total_counts,
            self.number_of_selections,
            average_reward
            )
        chosen_arm = np.argmax(ucb_values)

        self.number_of_selections[chosen_arm] += 1

        return chosen_arm

    def _factor_importance_each_arm(self, counts, num_selections, avg_reward):
        exploration_factor = np.sqrt(2 * np.log(counts) / num_selections)
        return avg_reward + exploration_factor

    def reward(self, chosen_arm):
        self.rewards[chosen_arm] += 1


class UCBTuned(UCB1):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def _factor_importance_each_arm(self, counts, num_selections, avg_reward):
        variance = (np.sum(np.square(self.rewards - avg_reward)))
        variance_factor = (1/num_selections) * variance

        tuned = np.sqrt(2 * np.log(counts) / num_selections)
        tuned_factor = variance_factor + tuned

        explo = np.minimum(1/4, tuned_factor)
        exploration_factor = np.sqrt((np.log(counts) / num_selections) * explo)

        return avg_reward + exploration_factor


class ThompsomSampling:
    def __init__(self, n_arms):
        self.number_reward_0 = np.zeros(n_arms).astype(np.float)
        self.number_reward_1 = np.zeros(n_arms).astype(np.float)

    def select(self):
        theta_value = np.random.beta(self.number_reward_1 + 1, self.number_reward_0 + 1)        
        chosen_arm = np.argmax(theta_value)        
        return chosen_arm
  
    def reward(self, chosen_arm):
        self.number_reward_1[chosen_arm] += 1
        
    def penalty(self, chosen_arm):
        self.number_reward_0[chosen_arm] += 1
