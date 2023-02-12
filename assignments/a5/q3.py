from rl.markov_process import MarkovProcess, NonTerminal, State
from rl.distribution import Distribution, Constant, Categorical, Gamma, Poisson, Gaussian, SampledDistribution

from rl.chapter9.order_book import OrderBook
from functools import partial
from dataclasses import dataclass
from numpy.random import poisson
from typing import Sequence
import itertools

@dataclass(frozen=True)
class DollarsAndShares:

    dollars: float
    shares: int

PriceSizePairs = Sequence[DollarsAndShares]

class OrderBookDynamics(MarkovProcess[OrderBook]):
    def __init__(self, sl_price_dist: Distribution[float], sl_num_dist: Distribution[int], 
    bl_price_dist: Distribution[float], bl_num_dist: Distribution[int], 
    sm_num_dist: Distribution[int], bm_num_dist: Distribution[int]):
        self.sl_price_dist = sl_price_dist # Price diff for sl  
        self.sl_num_dist = sl_num_dist # Num of sl order

        self.bl_price_dist = bl_price_dist # Price diff for bl  
        self.bl_num_dist = bl_num_dist # Num of bl order

        self.sm_num_dist = sm_num_dist # Num of sm order
        self.bm_num_dist = bm_num_dist # Num of bm order

    
    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[OrderBook]:
        def get_next_state(state: OrderBook):
            # do the market orders first
            next_state = state.sell_market_order(self.sm_num_dist.sample())[1]
            next_state = next_state.buy_market_order(self.bm_num_dist.sample())[1]
            if len(next_state.ascending_asks) == 0:
                price = next_state.descending_bids[0].dollars
            elif len(next_state.descending_bids) == 0:
                price = next_state.ascending_asks[0].dollars
            else:
                price = next_state.mid_price()
            # next do limit orders. First get quantity and then sample price within each
            for ord in range(self.sl_num_dist.sample()):
                new_price = max(price + self.sl_price_dist.sample(), 0)
                next_state = next_state.sell_limit_order(new_price, self.sl_num_dist.sample())[1]
            for ord in range(self.bl_num_dist.sample()):
                new_price = max(price + self.bl_price_dist.sample(), 0)
                next_state = next_state.buy_limit_order(new_price, self.bl_num_dist.sample())[1]
            return NonTerminal(next_state)
        func = partial(get_next_state, state.state)
        return SampledDistribution(func, expectation_samples=10000)

if __name__ == '__main__':
    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)

    print("Using Gaussian(0,2) for the price differential for sl and bl, Poisson(2) for the number of sl, bl, sm, bm")
    o = OrderBookDynamics(sl_price_dist=Gaussian(0,2), sl_num_dist=Poisson(2), 
    bl_price_dist=Gaussian(0,2), bl_num_dist=Poisson(2), 
    sm_num_dist=Poisson(2), bm_num_dist=Poisson(2))
    traces = o.traces(Constant(NonTerminal(ob0)))
    
    num_samples = 100
    samples = [list(itertools.islice(i, num_samples)) for i in itertools.islice(traces, 1)][0]
    # plot after 100 samples
    samples[-1].state.display_order_book()

    print("Using Gaussian(0,15) for the price differential for sl and bl, Poisson(2) for the number of sl, bl, sm, bm")
    o = OrderBookDynamics(sl_price_dist=Gaussian(0,15), sl_num_dist=Poisson(2), 
    bl_price_dist=Gaussian(0,15), bl_num_dist=Poisson(2), 
    sm_num_dist=Poisson(2), bm_num_dist=Poisson(2))
    traces = o.traces(Constant(NonTerminal(ob0)))
    
    num_samples = 100
    samples = [list(itertools.islice(i, num_samples)) for i in itertools.islice(traces, 1)][0]
    # plot after 100 samples
    samples[-1].state.display_order_book()

    print("Using Gamma(1, 3) for the price differential for sl and bl, Poisson(5) for the number of sl, bl, sm, bm")
    o = OrderBookDynamics(sl_price_dist=Gamma(1,3), sl_num_dist=Poisson(5), 
    bl_price_dist=Gamma(1,3), bl_num_dist=Poisson(5), 
    sm_num_dist=Poisson(5), bm_num_dist=Poisson(5))
    traces = o.traces(Constant(NonTerminal(ob0)))
    
    num_samples = 100
    samples = [list(itertools.islice(i, num_samples)) for i in itertools.islice(traces, 1)][0]
    # plot after 100 samples
    samples[-1].state.display_order_book()
    
        

