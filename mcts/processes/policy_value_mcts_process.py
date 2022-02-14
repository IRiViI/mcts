from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union, Any, List, Type

from mcts.nodes.policy_value_node import PolicyValueNode
from mcts.trees.policy_value_tree import PolicyValueTree
try:
    from torch import multiprocessing as mp
except ModuleNotFoundError:
    import multiprocessing as mp
import time

# class EnvironmentTree():

#     def __init__(self, original_env:Any, descending_env:Any, tree:PolicyValueTree):
#         self.original_env: Any = original_env
#         self.descending_env: Any = descending_env
#         self.tree: PolicyValueTree = tree

class MultiplayerMCTSSessionsProcess(mp.Process):

    def __init__(self, n_descent:int=1000, number_of_players:int=2, 
            init_temperature:int=1, min_temperature:int=0, steps_temperature:int=10, 
            cpuct:float=0.1, 
            values_lookup:dict={}, policies_lookup:dict={},
            batch_size:int=8, results_batch_size:int=64,
            seed:Union[int, None]=None,
            **kwargs):

        super(MultiplayerMCTSSessionsProcess, self).__init__(**kwargs)

        # Seed
        if seed is None:
            self.seed:int = np.random.randint(int(1e7))
        else:
            self.seed:int = seed
        
        # Some values
        self.n_descent:int = n_descent
        self.number_of_players:int = number_of_players
        self.cpuct:float = cpuct
        self.batch_size:int = batch_size
        self.results_batch_size:int = results_batch_size

        # Queues
        self.request_queue:mp.Queue = mp.Queue()
        self.response_queue:mp.Queue = mp.Queue()
        self.results_queue:mp.Queue = mp.Queue()
        self.resource_request_queue:mp.Queue=mp.Queue()
        self.resource_release_queue:mp.Queue=mp.Queue()

        # Temperature
        self.min_temperature:float = min_temperature
        self.init_temperature:float = init_temperature
        self.steps_temperature:float = steps_temperature
        self.temperature:np.ndarray = np.array([])

        # lookup tables
        self.values_lookup = values_lookup
        self.policies_lookup = policies_lookup

        # Trees
        self.trees: List[PolicyValueTree] = []
        self.leave_trees: List[PolicyValueTree] = []
        self.finished_trees: List[PolicyValueTree] = []
        self.queued_trees: List[PolicyValueTree] = []
        self.waiting_trees: List[PolicyValueTree] = []
        self.current_request: Any = None

        self.running:bool = False
        self.results:List[Any] = []

        self._has_resource_access = False
        self._resource_request_send = False

    # Process side
    def ask_resource_access(self) -> None:
        if self._resource_request_send is True: 
            raise RuntimeError("You may not send a second resource request")
        self._resource_request_send = True
        self.resource_request_queue.put(True)
        # print("put ask")

    def hold_ask_resource_access(self) -> None:
        while self._resource_request_send is True:
            time.sleep(0)
        self._resource_request_send = True
        self.resource_request_queue.put(True)
        # print("hold ask")

    def hold_until_resource_access(self) -> None:
        while self.i_has_resource_access():
            time.sleep(0)


    def has_pending_resource_request(self) -> bool:
        if self._resource_request_send is True:
            return True
        return False

    def i_has_resource_access(self) -> bool:
        # print(self._resource_request_send, self.resource_request_queue.qsize())
        if (self._resource_request_send is True) and (self.resource_request_queue.qsize() == 0):
            return True
        return False

    def release_resource(self) -> None:
        self._resource_request_send = False
        self.resource_release_queue.put(True) 
        # print("put release")
        
    # Other side
    def wants_resource_request(self) -> bool:
        if self.resource_request_queue.qsize() == 0:
            return False
        return True

    def grant_resource_access(self) -> None:
        if not self.wants_resource_request(): raise RuntimeError("Process does not want acces")
        self._has_resource_access = True
        self.resource_request_queue.get()
        # print("get request")

    def has_resource_access(self) -> bool:
        if self._has_resource_access:
            if self.resource_release_queue.qsize() == 0: return True
            self.resource_release_queue.get()
            self._has_resource_access = False
            # print("get release")
            return False
        return False


     # Done with that shizzle

    def clear_lookup_tables(self):
        self.policies_lookup.clear()
        self.values_lookup.clear()

    def temperature_step(self):
        self.temperature -= (self.init_temperature - self.min_temperature) / self.steps_temperature
        self.temperature[self.temperature <= self.min_temperature + 1e-8] = self.min_temperature

    def run(self):
        np.random.seed(self.seed)
        self.running = True

    def add_tree(self, tree: PolicyValueTree):
        self.trees.append(tree)

    def stop(self):
        self.running = False

    def step(self):
        for tree in [*self.trees]:
            # Check if the tree is currently descending
            if not tree.descending: continue
            if tree.is_finished:
                self.finished_trees.append(tree)
                self.trees.remove(tree)
                continue
                # raise RuntimeError("Tree was already finished")
            # Set step
            is_finished = tree.step()
            # Handle when tree is finished
            if is_finished:
                self.finished_trees.append(tree)
                self.trees.remove(tree)
                continue
            # Handle leave nodes
            if not tree.descending:
                self.leave_trees.append(tree)
                continue
            # # Handle regular in tree updates
            # self.updated_trees.append(tree)

    def get_leave_trees(self):
        leave_trees = [*self.leave_trees]
        del self.leave_trees[:]
        return leave_trees

    # def get_updated_trees(self):
    #     updated_trees = [*self.updated_trees]
    #     del self.updated_trees[:]
    #     return updated_trees

    # def add_to_waiting_list(self, tree: PolicyValueTree):
    #     self.waiting_trees.append(tree)

    def get_request_batch(self) -> List[PolicyValueTree]:
        # Only one batch at the time
        if len(self.waiting_trees) > 0: return []
        # Get next batch if any
        n_weighting = len(self.queued_trees)
        batch_size = np.min((self.batch_size, len(self.trees)))
        if n_weighting >= batch_size:
            self.waiting_trees = self.queued_trees[:batch_size]
            # Remove from waiting list
            del self.queued_trees[:batch_size]
            # Provide the batch
            return self.waiting_trees
        return []

    def get_results_batch(self) -> List[Any]:
        n_results = len(self.results)
        if n_results >= self.results_batch_size:
            results = [*self.results[:self.batch_size]]
            del self.results[:self.batch_size]
            return results
        return []

    def has_results(self) -> bool:
        if self.results_queue.qsize() == 0: return False
        return True

    def get_results(self):
        return self.results_queue.get()

    def put_results(self, results):
        self.results_queue.put(results)

    def queue_result(self, result:Any):
        self.results.append(result)

    def get_finished_trees(self) -> List[PolicyValueTree]:
        finished_trees = [*self.finished_trees]
        del self.finished_trees[:]
        return finished_trees

    def queue_request(self, tree:PolicyValueTree):
        # if not isinstance(tree, PolicyValueTree): raise ValueError("Tree must be of type Tree")
        self.queued_trees.append(tree)
        
    def has_request(self):
        if self.request_queue.qsize() == 0: return False
        return True

    def put_request(self, request):
        self.request_queue.put(request)
        self.current_request = request

    def get_request(self):
        return self.request_queue.get()

    def put_response(self, batch):
        self.response_queue.put(batch)

    def has_response(self) -> bool:
        if self.response_queue.qsize() == 0: return False
        return True

    def get_response(self):
        response = self.response_queue.get()
        waiting_trees = [*self.waiting_trees]
        request = self.current_request
        if waiting_trees is None:
            raise RuntimeError("No batch request found")
        if len(waiting_trees) == len(response):
            pass
        elif len(waiting_trees) == len(response[0]):
            pass 
        else:
            raise RuntimeError(f"Length of batch ({len(waiting_trees)}) and response ({len(response)}) do no match.")
        del self.waiting_trees[:]
        self.current_request = None
        return response, request, waiting_trees

    # def has_new_environments(self) -> bool:
    #     if self.env_queue.qsize() == 0: return False
    #     return True

    # def put_new_environments(self, state):
    #     self.env_queue.put(state)

    # def get_new_environments(self):
    #     return self.env_queue.get()