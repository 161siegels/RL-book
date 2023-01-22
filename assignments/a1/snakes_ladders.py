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


class SnakesLaddersMPFinite(FiniteMarkovProcess[GameState]):

    def __init__(
        self,
        capacity: int,
        dice_sides: int,
        snakes: Dict[GameState, GameState],
        ladders: Dict[GameState, GameState]
    ):
        self.capacity: int = capacity
        self.dice_sides: int = dice_sides,
        self.snakes: Dict = snakes
        self.ladders: Dict = ladders

        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[GameState, FiniteDistribution[GameState]]:
        d: Dict[GameState, Categorical[GameState]] = {}
        # last square is terminal so do not worry about it
        for alpha in range(self.capacity):
            state = GameState(alpha)
            probs = np.zeros(self.capacity+1)
            if state in self.snakes:
                probs[self.snakes[state].square_num] = 1
            elif state in self.ladders:
                probs[self.ladders[state].square_num] = 1
            elif alpha!=self.capacity:
                # do not allow transitions to states greater than capcity
                for i in range(alpha+1, min(self.capacity+1, alpha+dice_sides+1)):
                    probs[i] = 1/dice_sides
                # probs[alpha] = 1-probs.sum()
                probs[100] = 1-probs[:100].sum()
            all_states = [GameState(x) for x in np.arange(0, self.capacity+1)]
            d[state] = Categorical(dict(zip(all_states, probs)))
        return d

if __name__ == '__main__':
    capacity = 100
    dice_sides = 6
    ladders = {GameState(1):GameState(38), GameState(4):GameState(14), GameState(8):GameState(30),
    GameState(21):GameState(42), GameState(28):GameState(76), GameState(50):GameState(67),
    GameState(71):GameState(92), GameState(80):GameState(99)}

    snakes = {GameState(32):GameState(10), GameState(36):GameState(6), GameState(48):GameState(26),
    GameState(62):GameState(18), GameState(88):GameState(24), GameState(95):GameState(56),
    GameState(97):GameState(78)}

    sl_mp = SnakesLaddersMPFinite(
        capacity=capacity,
        dice_sides=dice_sides,
        ladders=ladders,
        snakes=snakes
    )

    print("Transition Map")
    print("--------------")
    start_state = Constant(NonTerminal(GameState(0)))
    traces = sl_mp.traces(start_state)
    
    num_samples = 100
    samples = [list(itertools.islice(i, 100000)) for i in itertools.islice(traces, num_samples)]
    plot_traces = [[(step_num, s.state.square_num) for step_num, s in enumerate(sample)] for sample in samples]

    plt.xlabel("Step Num")
    plt.ylabel("Square Num")
    plt.title("Sample Traces")
    for i in range(10):
        plt.plot([pt[0] for pt in plot_traces[i]],[pt[1] for pt in plot_traces[i]],label = 'Sample %s'%i)
    plt.legend()
    plt.savefig("assignments/a1/sample_traces.png")
    plt.clf()

    total_steps = [len(s) for s in plot_traces]
    plt.hist(total_steps, density=True, bins=50)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Steps to Completion')
    plt.title("Histogram of Num Steps")
    plt.savefig("assignments/a1/histogram.png")


    print("Stationary Distribution")
    print("-----------------------")
    trans_reward_map = {}
    snake_ladder_starts = list(snakes.keys()) + list(ladders.keys())
    for ss, dist in sl_mp.transition_map.items():
        reward = 0 if (ss.state in snake_ladder_starts) else 1
        trans_reward_map[ss] = Categorical({(k, reward):v for k, v in dict(dist).items()})


    fmrp = FiniteMarkovRewardProcess(trans_reward_map)
    print("Expected Number of steps: ", fmrp.get_value_function_vec(gamma=1)[0])
