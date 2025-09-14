from __future__ import annotations
from typing import Dict, List, Tuple
import math
import numpy as np

from policy import Policy
from mdp import MDP, State, Action
from mdp_utils import enumerate_states
from collections import deque

try:
    from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
except Exception:
    UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class MyPolicy(Policy):
    """
    Value-free constructive policy based on shortest-path over the MDP's
    most-likely-successor graph (no v^pi, no returns, no evaluation calls).

    Steps:
      1) Enumerate reachable states.
      2) Build a directed graph: for each (state, action), connect to the
         most likely successor under mdp.transition(s, a).
      3) Reverse-BFS from all goals to compute a discrete distance d(s).
      4) For each non-terminal state s, choose the action minimizing d(next).
         Tie-break with a fixed action order.

    Notes:
      • Holes are treated as terminals and are not seeded in BFS, so d(H)=∞,
        which naturally discourages stepping into holes unless unavoidable.
      • Absorbing and goals get d=0.
    """

    def __init__(
        self,
        mdp: MDP,
        rng: np.random.Generator,
        tie_break: Tuple[Action, ...] = (RIGHT, DOWN, LEFT, UP),
    ):
        super().__init__(mdp, rng)
        self.mdp = mdp
        self.rng = rng
        self.tie_break = tie_break
        self._states: List[State] = []
        self._state_idx: Dict[State, int] = {}
        self._policy: Dict[State, Action] = {}
        self._build()


    def _decision(self, s: State) -> Action:
        return self._policy.get(s, ABSORB)        

    def _most_likely_successor(self, s: State, a: Action) -> State:
        succs = self.mdp.transition(s, a)
        if succs:
            return max(succs, key=lambda x: x[1])[0]
        return s 

    def _build(self) -> None:
        self._states = list(enumerate_states(self.mdp))
        self._policy: Dict[State, Action] = {}

        d = {s: math.inf for s in self._states}
        queue = deque()

        for s in self._states:
            if s[1] == "G" or s[1] == "⊥":
                d[s] = 0
                queue.append(s)
            
        while queue:
            s_next = queue.popleft()
            for s in self._states:
                for a in [UP, RIGHT, DOWN, LEFT]:
                    if self.mdp.is_terminal(s):
                        continue
                    if self._most_likely_successor(s, a) == s_next:
                        if d[s] > d[s_next] + 1:
                            d[s] = d[s_next] + 1
                            queue.append(s)

        for s in self._states:
            if self.mdp.is_terminal(s):
                continue
            best_a = None
            best_d = math.inf
            for a in self.mdp.actions(s):
                s_next = self._most_likely_successor(s, a)

                if s_next[1] == "H":
                    candidate_d = math.inf
                else:
                    candidate_d = d[s_next]

                if candidate_d < best_d or (
                    candidate_d == best_d
                    and (best_a is None or self.tie_break.index(a) < self.tie_break.index(best_a))
                ):
                    best_d = candidate_d
                    best_a = a

            if best_a is not None:
                self._policy[s] = best_a
