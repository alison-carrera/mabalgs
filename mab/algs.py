import numpy as np

class UCB1:
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
       
        ucb_values = self._factor_importance_each_arm(total_counts, self.number_of_selections, average_reward)
        chosen_arm = np.argmax(ucb_values)
        
        self.number_of_selections[chosen_arm] += 1
        
        return chosen_arm
        
    def _factor_importance_each_arm(self, total_counts, number_of_selections, average_reward):
        exploration_factor = np.sqrt(2 * np.log(total_counts) / number_of_selections)
        return average_reward + exploration_factor
        
    def reward(self, chosen_arm):
        self.rewards[chosen_arm] += 1