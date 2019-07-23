import numpy as np
import random

"""

    This class contains another research field for MAB algorithms. The ranked MAB.

    Papers: [1] Learning Diverse Rankings with Multi-Armed Bandits.

    @author Alison Carrera

"""


class RBA:

    """

        This class represents a ranked mab algorithm implementation by paper [1]

    """

    def __init__(self, n_arms, n_ranks, mab_algorithm_class):
        """
            Constructor for RBA.

            :param n_arms: Number of arms that RBA will perform on.
            :param n_ranks: Number of rank position that RBA will perform on.
            :param mab_algorithm_class: Class reference of MAB algorithm which will be used.

        """
        self.n_ranks = n_ranks
        self.mab_alg_type = mab_algorithm_class
        self.original_arms = set(list(range(n_arms)))
        self.ranks = []

        for i in range(n_ranks):
            self.ranks.append(mab_algorithm_class(n_arms))

    def resolve_conflict(self, selected_arms, optional_elements):
        """
            This method is responsible to resolve conflicts between mabs in all ranks.

            :param selected_arms: Selected arms chosen by the RBA.
            :param optional_elements: This can be used to pass any extra value that you need to resolve conflicts.

            :return: Return the select arm after conflict handle.
        """
        available_arms = self.original_arms - set(selected_arms)
        return random.choice(list(available_arms))

    def select(self):
        """
            This method selects the arms for all ranks.

            :return: Return selected arm for every rank.
        """

        selected_arms = []

        for i in range(self.n_ranks):
            selected_arm, ranked_arms = self.ranks[i].select()

            if selected_arm in selected_arms:
                selected_arms.append(self.resolve_conflict(
                    selected_arms, ranked_arms))
            else:
                selected_arms.append(selected_arm)

        return selected_arms

    def reward(self, selected_arms, chosen_arm):
        """
            This method is responsible to reward the chosen arm in RBA.
        """

        for arm in selected_arms:
            rank_index = selected_arms.index(arm)
            if arm == chosen_arm:
                self.ranks[rank_index].reward(arm)


class RBAM(RBA):

    """

        This class represents a modified ranked mab algorithm from [1].

        We have implemented our own colision method for RBA. It perform better than RBA when
        the arms weights in a same position of the ranked are very near each other.

        When this occurs RBA choose arms in position using a random way. 
        We simple get the best following non used arm from MAB result instead of using a random way.
        With this simple modification we have got a better results.  

    """

    def __init__(self, n_arms, n_ranks, mab_algorithm_class):
        """
            Constructor for RBAM.

            :param n_arms: Number of arms that RBAM will perform on.
            :param n_ranks: Number of rank position that RBAM will perform on.
            :param mab_algorithm_class: Class reference of MAB algorithm which will be used.

        """
        super().__init__(n_arms, n_ranks, mab_algorithm_class)

    def resolve_conflict(self, selected_arms, ranked_arms):
        """
            If ranked_arms[0] is in selected_arms, instead return a random arm from ranked_arms 
            excluding the position[0] (like RBA), we get the next better element 
            from the arm (given by MAB) which is ranked_arms[1] and so on.

            :param selected_arms: Selected arms from RBAM in some iteration.
            :param ranked_arms: Ranked arms for that rank position.
            :return: An arm.

        """
        for ranked_value in ranked_arms:
            if ranked_value not in selected_arms:
                return ranked_value
