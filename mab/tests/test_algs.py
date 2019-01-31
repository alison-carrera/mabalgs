from mab import algs
import numpy as np


def test_ucb_init_return_first_arm():
    ucb_with_two_arms = algs.UCB1(2)
    assert ucb_with_two_arms.select()[0] == 0


def test_ucb_use_all_arm_dont_usage():
    ucb_with_two_arms = algs.UCB1(2)
    assert ucb_with_two_arms.select()[0] == 0
    assert ucb_with_two_arms.select()[0] == 1


def test_ucb_use_all_arm_dont_usage_after_priorize():
    ucb_with_two_arms = algs.UCB1(2)
    assert ucb_with_two_arms.select()[0] == 0
    assert ucb_with_two_arms.select()[0] == 1
    assert ucb_with_two_arms.select()[0] == 1


def test_ucb_select_two_arms_and_success_return_second():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.select()
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    assert ucb_with_two_arms.select()[0] == 1


def test_ucb_select_two_arms_and_success_one_return_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.select()
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    assert ucb_with_two_arms.select()[0] == 0


def test_ucb_select_two_arms_and_have_two_reward_priorize_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    assert ucb_with_two_arms.select()[0] == 1


def test_ucb_exploration_first():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    last_arm = ucb_with_two_arms.select()[0]
    assert last_arm == 1


def test_ucb_exploration_second():
    ucb_with_two_arms = algs.UCB1(2)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(0)
    ucb_with_two_arms.select()
    ucb_with_two_arms.reward(1)
    ucb_with_two_arms.select()
    last_arm = ucb_with_two_arms.select()[0]
    assert last_arm == 0
