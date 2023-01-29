from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import itertools



@dataclass(frozen=True)
class EmploymentState():
    employed: bool
    job_idx: int

class EmploymentWages():
    def __init__(self, job_offer_probs: List[float], 
    job_wages: List[float],
    unemployment_wage: float,
    gamma: float, 
    fired_prob: float,
    tol: float = 1e-9):
        self.job_offer_probs: List[float] = job_offer_probs
        self.wages: List[float] = [unemployment_wage] + job_wages
        self.gamma: float = gamma
        self.fired_prob: float = fired_prob
        self.tol = tol
        self.value_func: Mapping[EmploymentState] = {EmploymentState(i, 1+j): 0.0 for i, j in itertools.product([True, False], range(len(job_wages)))}
        self.check_inputs()
    
    def check_inputs(self):
        assert(len(self.job_offer_probs) == (len(self.wages)-1))
        # job offer probs sum to 1
        assert(abs(1 - sum(self.job_offer_probs)) <= 1e-10)
        # all wages are non-negative
        assert(sum([w>0 for w in self.wages]) == len(self.wages))
        assert((self.gamma>=0) & (self.gamma<=1))
        assert((self.fired_prob>=0) & (self.fired_prob<=1))
        assert(self.tol>0)

    def opt_value_func(self):
        utility = [np.log(w) for w in self.wages]
        convergence = 1
        while convergence > self.tol:
            old_vf = self.compute_value_func(utility = utility)
            # max dif in value func
            convergence = max([abs(old_vf[i] - v) for i, v in self.value_func.items()])
        return self.value_func
    
    def compute_value_func(self, utility: List[float]):
        old_vf = self.value_func.copy()
        # old value function for the employed states
        employed_vf = [old_vf[EmploymentState(True, i+1)] for i in range(len(self.job_offer_probs))]
        # old value function for the unemployed states
        unemployed_vf = [old_vf[EmploymentState(False, i+1)] for i in range(len(self.job_offer_probs))]
        # value if decline job offer in unemployed state
        decline_vf = utility[0] + self.gamma * sum(np.array(unemployed_vf)*np.array(self.job_offer_probs))

        for i in range(len(self.job_offer_probs)):
            # take max of if decline or accept offer when unemployed
            self.value_func[EmploymentState(False, i+1)] = max(employed_vf[i], decline_vf)
            self.value_func[EmploymentState(True, i+1)] = utility[i+1] + self.gamma * (
                (1-self.fired_prob)*old_vf[EmploymentState(True, i+1)] + self.fired_prob*sum(np.array(unemployed_vf)*np.array(self.job_offer_probs)))
        return old_vf
    
    def optimal_policy(self):
        policy = []
        for i in range(len(self.job_offer_probs)):
            if abs(self.value_func[EmploymentState(False, i+1)] - self.value_func[EmploymentState(True, i+1)]) <= 1e-4:
                policy.append("Accept")
            else:
                policy.append("Decline")
        return policy


if __name__ == '__main__':
    # job offers probabilities
    job_offer_probs = [0.25, 0.25, 0.25, 0.25]
    # wages for each of the jobs
    job_wages =  [6, 7, 8, 9]
    unemployment_wage = 5
    gamma = 0.8
    fired_prob = 0.05

    ew = EmploymentWages(
        job_offer_probs=job_offer_probs,
        job_wages=job_wages,
        unemployment_wage = unemployment_wage,
        gamma=gamma,
        fired_prob=fired_prob,
        tol=0.000001
    )
    opt_vf = ew.opt_value_func()
    unemployment_str = "Unemployed with offer from job"
    employment_str = "Employed at job "
    for k, v in opt_vf.items():
        print(f"{employment_str if k.employed else unemployment_str} {k.job_idx}: {v}")
    print("Optimal Policy: ", ew.optimal_policy())
    
