from rl.distribution import SampledDistribution, Distribution
from rl.markov_decision_process import MarkovDecisionProcess, NonTerminal, State, Terminal
from rl.policy import DeterministicPolicy
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    ValueFunctionApprox
from dataclasses import dataclass
from typing import TypeVar, Iterable, Callable, Sequence, Tuple, Iterator
from rl.distribution import Gaussian


A = TypeVar('A')
S = TypeVar('S')

class OptimalOptionsExecution:
    '''

    time_steps refers to the number of time steps T.

    avg_exec_price_diff refers to the time-sequenced functions g_t
    that gives the average reduction in the price obtained by the
    Market Order at time t due to eating into the Buy LOs. g_t is
    a function of PriceAndShares that represents the pair of Price P_t
    and MO size N_t. Sales Proceeds = N_t*(P_t - g_t(P_t, N_t)).

    price_dynamics refers to the time-sequenced functions f_t that
    represents the price dynamics: P_{t+1} ~ f_t(P_t, N_t). f_t
    outputs a distribution of prices.

    utility_func refers to the Utility of Sales proceeds function,
    incorporating any risk-aversion.

    discount_factor refers to the discount factor gamma.

    func_approx refers to the FunctionApprox required to approximate
    the Value Function for each time step.

    initial_price_distribution refers to the distribution of prices
    at time 0 (needed to generate the samples of states at each time step,
    needed in the approximate backward induction algorithm).
    '''
    def __init__(self,
    time_steps: int,
    price_dynamics: Sequence[Callable[[float], Distribution[float]]],
    utility_func: Callable[[float], float],
    discount_factor: float,
    func_approx: ValueFunctionApprox[float],
    initial_price_distribution: Distribution[float],
    strike_price: float,
    # if false then put option, else call option
    call_option: bool):
        self.time_steps: int = time_steps
        self.price_dynamics: Sequence[Callable[[float], Distribution[float]]] = price_dynamics
        self.utility_func: Callable[[float], float] = utility_func
        self.discount_factor: float = discount_factor
        self.func_approx: ValueFunctionApprox[float] = func_approx
        self.initial_price_distribution: Distribution[float] = \
            initial_price_distribution
        self.strike_price: float = strike_price
        self.call_option: bool = call_option


    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, int]:
        utility_f: Callable[[float], float] = self.utility_func
        price_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.price_dynamics
        steps: int = self.time_steps
        strike_price = self.strike_price
        class OptimalBinomialTree(MarkovDecisionProcess[float, str]):
            def __init__(self):
                self.strike_price = 100
            
            def step(
                self,
                state: NonTerminal[float],
                action: bool
            ) -> Distribution[Tuple[State[float], float]]:
                    def state_reward_sampler_func(
                            prev_price=state.state
                        ) -> Tuple[State[float], float]:
                            next_price: float = price_dynamics[t](prev_price).sample()
                            if action:
                                # reward for a call option
                                reward: float = (strike_price - next_price) if call_option else (next_price - strike_price)
                                reward: float = utility_f(reward)
                                next_state: State[float] = Terminal(next_price)
                            else:
                                reward: float = utility_f(0)
                                next_state: State[float] = NonTerminal(next_price)
                            return (next_state, reward)
                    return SampledDistribution(
                            sampler=state_reward_sampler_func,
                            expectation_samples=100
                        )
            def actions(self, state: NonTerminal[S]) -> Iterable[A]:
                return [True, False]

        return OptimalBinomialTree()

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[float]]:

        def states_sampler_func() -> NonTerminal[float]:
            price: float = self.initial_price_distribution.sample()
            for i in range(t):
                price = self.price_dynamics[i](price).sample()
            return NonTerminal(price)

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(
        self
    ) -> Iterator[Tuple[ValueFunctionApprox[float],
                        DeterministicPolicy[float, bool]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, bool],
            ValueFunctionApprox[float],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            self.func_approx,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps)]

        num_state_samples: int = 5000
        error_tolerance: float = 1e-3

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.discount_factor,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':
    # choosing random parameters as test case
    init_price_mean: float = 100.0
    init_price_stdev: float = 10.0
    num_shares: int = 100
    num_time_steps: int = 5
    alpha: float = 0.03
    beta: float = 0.05
    strike_price: float = 100.0
    call_option: bool = True
    
    dynamics = [lambda p_s: Gaussian(
        μ=p_s,
        σ=0.
    ) for _ in range(num_time_steps)]
    ffs = [
        lambda p_s: p_s.state,
        # lambda p_s: float(p_s.state* p_s.state)
    ]
    fa: FunctionApprox = LinearFunctionApprox.create(feature_functions=ffs)
    init_price_distrib: Gaussian = Gaussian(
        μ=init_price_mean,
        σ=init_price_stdev
    )

    ooe: OptimalOptionsExecution = OptimalOptionsExecution(
        time_steps=num_time_steps,
        price_dynamics=dynamics,
        utility_func=lambda x: x,
        discount_factor=1,
        func_approx=fa,
        initial_price_distribution=init_price_distrib,
        strike_price=strike_price,
        call_option=call_option
    )
    it_vf: Iterator[Tuple[ValueFunctionApprox[float],
                          DeterministicPolicy[float, int]]] = \
        ooe.backward_induction_vf_and_pi()

    state: float =init_price_mean
    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()
    for t, (vf, pol) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()
        exercise: bool = pol.action_for(state)
        val: float = vf(NonTerminal(state))
        print(f"Optimal to Exercise = {exercise}, Opt Val = {val:.3f}")
        print()
        print("Optimal Weights below:")
        print(vf.weights.weights)
        print()
