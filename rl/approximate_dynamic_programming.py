from typing import Iterator, Mapping, TypeVar

from rl.function_approx import FunctionApprox
from rl.iterate import converged, iterate
from rl.markov_process import FiniteMarkovRewardProcess

S = TypeVar('S')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


def evaluate_finite_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        gamma: float,
        approx_0: FunctionApprox[S]) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the value function for the give Markov reward
    process, using the given FunctionApprox to approximate the value
    function at each step.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        vs = v.evaluate(mrp.non_terminal_states)
        updated = mrp.reward_function_vec + gamma *\
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.states(), updated))

    return iterate(update, approx_0)


def evaluate_mrp(
        mrp: MarkovRewardProcess[S],
        γ: float,
        approx_0: FunctionApprox[S]) -> Iterator[FunctionApprox[S]]:

    '''Iteratively calculate the value function for the give Markov reward
    process, using the given FunctionApprox to approximate the value
    function at each step for a random sample of the process's states.

    '''
    def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
        vs = v.evaluate(mrp.sample_states())
        updated = mrp.reward_function_vec + gamma *\
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.states(), updated))

    return iterate(update, approx_0)
