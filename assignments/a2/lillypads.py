from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical, Constant, FiniteDistribution
from rl.markov_process import NonTerminal
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from scipy.stats import poisson
import numpy as np
import itertools
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GameState:
    square_num: int

def get_transition_reward_map(capacity=10):
    trans_reward_map: Dict[GameState, Categorical[(GameState, float)]] = {}
    for i in range(capacity):
        state = GameState(i)
        probs = np.zeros(capacity+1)
        probs[i+1:] = 1/(capacity-i)
        keys = [(GameState(x), 1) for x in np.arange(0, capacity+1)]
        trans_reward_map[state] = Categorical(dict(zip(keys, probs)))
    return trans_reward_map

if __name__ == '__main__':
    trans_reward_map = get_transition_reward_map()
    fmrp = FiniteMarkovRewardProcess(trans_reward_map)
    print("Expected Number of steps: ", fmrp.get_value_function_vec(gamma=1)[0])