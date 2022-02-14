from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union
from matplotlib import pyplot as plt
import time

if TYPE_CHECKING:
    from mcts.nodes.policy_value_node import PolicyValueNode
    

class PolicyValueNode():

    counter = 0

    def __init__(self, values, policy, actions, player, num_of_players, termination,
                string_notation:str=None, parent=None, annotation:str=None):
        
        if len(policy) != len(actions):
            raise f"The length of the policy ({len(policy)}) must be the same a the number of actions ({len(actions)})"
        if not (player >= 0 and player < num_of_players):
            raise f"player number {player} exceeds the max number of players"
        self.values = values
        self.string_notation = string_notation
        self.policy = policy
        self.actions = [*actions]
        self.player = player
        self.termination = termination

        self.level = 0
        self.index = PolicyValueNode.counter
        PolicyValueNode.counter+=1
        self.annotation = f"{self.index}"
        if annotation:
            self.annotation = annotation
        elif self.string_notation:
            self.annotation = self.string_notation

        # for action in self.actions:
        #     if isinstance(action,np.ndarray):
        #         # Atm I'm just to lazy to handle this case
        #         raise ValueError("Action may not be numpy arrays")

        self.number_of_edges = len(policy)
        
        self.qs = np.zeros((num_of_players, self.number_of_edges), dtype=float)
        self.ws = np.zeros((num_of_players, self.number_of_edges), dtype=float)
        self.ns = np.zeros(self.number_of_edges, dtype=float)

        self.parent = parent
        self.children = [None for _ in range(self.number_of_edges)]

        self.tree: Union[PolicyValueTree, None] = None

    def u(self):
        if self.tree is None: 
            raise RuntimeError("No tree")
        # Add dirichlet noise to the root node
        if self.parent == None and self.tree.dirichelt_epsilon > 0:
            policy = (1 - self.tree.dirichelt_epsilon) * self.policy + self.tree.dirichelt_epsilon * np.random.dirichlet(self.policy)
        else:
            policy = self.policy
        return self.tree.cpuct * policy * np.sqrt(self.n()) / (1 + self.ns)

    def q(self):
        return self.qs[self.player]

    def n(self):
        return np.sum(self.ns)

    def accumulated_policy(self):
        return self.ns / np.sum(self.ns)
    
    def visit(self, node, values):
        index = self.children.index(node)
        self.ns[index] += 1
        self.ws[:, index] += values
        self.qs[:, index] = self.ws[:, index] / self.ns[index]
        if self.parent:
            self.parent.visit(self, values)

    # def set_child(self, child, action):
    #     match = False
    #     for index, _action in enumerate(self.actions):
    #         if _action == action:
    #             match = True
    #             break
    #     if match == False:
    #         raise ValueError("Action could not be found. Maybe it is not in the corect form?")
    #     if self.children[index] != None:
    #         raise RuntimeError("Child has already been set")
    #     self.children[index] = child
                