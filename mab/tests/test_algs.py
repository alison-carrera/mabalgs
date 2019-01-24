from mab import algs
import numpy as np

def test_ucb_init_return_first_arm():
    ucb_with_two_arms = algs.UCB1(2)
    assert ucb_with_two_arms.select_arm() == 0

def test_ucb_select_two_arms_and_success_return_second():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.update(0, 0)
    ucb_with_two_arms.update(1, 1)
    assert ucb_with_two_arms.select_arm() == 1

def test_ucb_select_two_arms_and_success_one_return_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(1, 0)
    assert ucb_with_two_arms.select_arm() == 0

def test_ucb_select_two_arms_and_have_two_reward_priorize_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(1, 1)
    ucb_with_two_arms.update(1, 1)
    assert ucb_with_two_arms.select_arm() == 0

def test_ucb_exploration_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(1, 1)
    ucb_with_two_arms.update(1, 1)
    ucb_with_two_arms.update(1, 0)
    assert ucb_with_two_arms.select_arm() == 0

def test_ucb_exploration_second():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(0, 1)
    ucb_with_two_arms.update(0, 0)
    ucb_with_two_arms.update(1, 1)
    ucb_with_two_arms.update(1, 1)
    assert ucb_with_two_arms.select_arm() == 1
