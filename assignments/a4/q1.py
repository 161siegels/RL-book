
from typing import Iterator, Tuple, TypeVar, Sequence, Mapping
import numpy as np
from rl.approximate_dynamic_programming import value_iteration

from rl.function_approx import Dynamic, Tabular
from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
from rl.approximate_dynamic_programming import evaluate_mrp
from rl.iterate import converged, iterate
from rl.markov_process import (MarkovRewardProcess,
                               NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess
                                        )
from rl.policy import Policy, DeterministicPolicy
from rl.distribution import Categorical
from rl.distribution import Choose
from rl.dynamic_programming import policy_iteration_result

S = TypeVar('S')
A = TypeVar('A')
V = Mapping[NonTerminal[S], float]
NTStateDistribution = Distribution[NonTerminal[S]]
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]





def approx_policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_v_0: FunctionApprox[S],
    pol: DeterministicPolicy[S, A],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[Tuple[FunctionApprox[S], DeterministicPolicy[S, A]]]:
    def update(vf_policy: Tuple[ValueFunctionApprox[S], Policy[S, A]]) \
            -> Tuple[FunctionApprox[S], DeterministicPolicy[S, A]]:

        nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(num_state_samples)

        vf, pi = vf_policy
        mrp: MarkovRewardProcess[S] = mdp.apply_policy(pi)
        # update value function
        updated_vf: FunctionApprox[S] = converged(
            evaluate_mrp(mrp, γ, vf, non_terminal_states_distribution, num_state_samples),
            done=lambda a, b: a.within(b, 1e-4)
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + γ * updated_vf.evaluate([s1]).item()
        # get optimal action for each state
        def optimal_action(s: S) -> A:
            a = max(
            ((mdp.step(NonTerminal(s), a).expectation(return_), a)
             for a in mdp.actions(NonTerminal(s)))
        )[1]
            return a

        # return tuple of update vf and policy
        updated_vf = updated_vf.update([(s, max(mdp.step(s, a).expectation(return_)
                                       for a in mdp.actions(s))) for s in nt_states])
        updated_policy = DeterministicPolicy(optimal_action)
        return (updated_vf, updated_policy)


    return iterate(update, (approx_v_0, pol))




if __name__ == '__main__':
    # construct fake example
    non_terminal_states = [0, 1, 2]
    terminal_state = [3]
    all_states = non_terminal_states+terminal_state
    actions = [0, 1]
    mdp_map = {}
    for s in non_terminal_states:
        mdp_map[s] = {}
        for a in actions:
            probs = np.random.rand(len(all_states))
            probs = probs/probs.sum()
            mdp_map[s][a] = Categorical(dict(zip([(s,(1 if s in terminal_state else 0)) for s in all_states], probs)))

    this_mdp = FiniteMarkovDecisionProcess(mdp_map)
    start = Dynamic({NonTerminal(s): 0.0 for s in
                         non_terminal_states})
    dist = Choose([NonTerminal(s) for s in non_terminal_states])

    # Test approx value iteration
    value_iter_result = converged(
            value_iteration(
    mdp=this_mdp,
    γ = 1,
    approx_0 = start,
    non_terminal_states_distribution = dist,
    num_state_samples = 10
),
            done=lambda a, b: a.within(b, 1e-6)
        )
    # Test approx policy iteration
    def start_actions(s: S) -> A:
        return 0
    policy_iter_result = converged(
            approx_policy_iteration(
    mdp=this_mdp,
    γ = 1,
    approx_v_0 = start,
    pol=DeterministicPolicy(start_actions),
    non_terminal_states_distribution = dist,
    num_state_samples = 10
),
            done=lambda a, b: a[0].within(b[0], 1e-6)
        )
    # First, compare approx value iter with approx polciy iter
    print(f"approx value iteration optimal value function: {value_iter_result}")
    print(f"approx policy iteration optimal value function: {policy_iter_result[0]}")

    # Make sure max difference in opt value function is less than threshold
    max_val_dif = max([abs(value_iter_result.values_map[NonTerminal(s)] - (policy_iter_result[0].values_map[NonTerminal(s)])) for s in non_terminal_states])
    assert(max_val_dif < 1e-4)

    # Next, compare approx policy iter with exact polciy iter
    exact_policy_iter = policy_iteration_result(this_mdp, gamma = 1)
    print(f"exact policy iteration optimal value function: {policy_iter_result[0]}")
    # Make sure max difference in opt value function is less than threshold
    max_val_dif = max([abs(exact_policy_iter[0][NonTerminal(s)] - (policy_iter_result[0].values_map[NonTerminal(s)])) for s in non_terminal_states])
    assert(max_val_dif < 1e-4)

    # Shows the policies are same too
    print(f"approx policy iteration optimal policy: {[f'For state {s}: do Action {policy_iter_result[1].act(NonTerminal(s)).value}' for s in range(3)]}")
    print(f"exact policy iteration optimal policy: {exact_policy_iter[1]}")

    