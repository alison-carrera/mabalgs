import numpy as np

class UCB1:
    def __init__(self, n_arms):
        self.number_of_selections = np.zeros(n_arms).astype(np.float)
        self.average_reward = np.zeros(n_arms).astype(np.float)
        self.rewards = np.zeros(n_arms).astype(np.float)
        self.n_arms = n_arms

    def select_arm(self):
        temp = np.where(self.number_of_selections == 0)[0]
        if len(temp) > 0:
            return temp[0]
       
        ucb_values = np.zeros(self.n_arms).astype(np.float)
        total_counts = np.sum(self.number_of_selections)
       
        bonus = np.sqrt((2 * np.log(total_counts)) / self.number_of_selections)
        ucb_values = self.average_reward + bonus
        return np.argmax(ucb_values)
  
    def update(self, chosen_arm, reward):
        self.number_of_selections[chosen_arm] = self.number_of_selections[chosen_arm] + 1
        self.rewards[chosen_arm] = self.rewards[chosen_arm] + reward
        self.average_reward[chosen_arm] = self.rewards[chosen_arm] / self.number_of_selections[chosen_arm]