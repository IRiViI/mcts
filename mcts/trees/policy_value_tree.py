from __future__ import annotations
import numpy as np
from typing import Union, Dict, TYPE_CHECKING, List
import networkx as nx
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from mcts.nodes.policy_value_node import PolicyValueNode

    

class PolicyValueTree():

    def __init__(self, 
        cpuct=0.3, dirichelt_epsilon:float=0.25, n_descent=100,
        shared_values=None, shared_policies=None):

        self.nodes = []

        self.cpuct = cpuct
        self.dirichelt_epsilon:float = dirichelt_epsilon
        self.n_descent = n_descent

        self.current_node = None

        # Holds the values belonging to string notation
        self.values = {} if shared_values == None else shared_values
        self.policies = {} if shared_policies == None else shared_policies

        self.is_finished = False
        self.descending = True
        self.current_action_index = -1

        # Ugly temporary solution?
        self.storage: Dict = {}

    @property
    def root_node(self) -> PolicyValueNode:
        return self.nodes[0]

    def get_string_notation(self):
        if self.current_node is None: raise RuntimeError("No current node")
        return self.current_node.string_notation

    def get_action(self):
        node = self.current_node
        if node is None: raise RuntimeError("No current node")
        index = np.argmax(node.q() + node.u())
        return (node.actions[index]), index, node.children[index]

    def step(self) -> bool:
        """[summary]

        Returns:
            bool: If the tree is finished
        """        
        if self.current_node is None:
            self.descending = False
            self.current_action_index = -1
            return False
        action, self.current_action_index, node = self.get_action()
        # Check if it's a leave node
        if node == None:
            self.descending = False
            return False
        elif node.termination:
            # Handle termination nodes
            self.current_node.visit(node, node.values)
            self.current_node = self.root_node
        else:
            # Set the next node as the current node
            self.current_node = node
        return self.check_if_finished()

    def check_if_finished(self):
        self.is_finished = self.root_node.n() >= self.n_descent
        if self.is_finished:
            self.descending = False
            self.current_action_index = -1
        return self.is_finished

    def get_policy_and_value(self, string_notation):
        if string_notation in self.values and string_notation in self.policies:
            return self.policies[string_notation], self.values[string_notation]
        return None, None

    def add_node(self, action_index, node):
        assert self.descending == False, "Tree is still descending"
        assert node.tree is None, "Node alrady has a tree"
        root: bool = False
        if self.current_node is not None:
            assert action_index != -1, "Action index should not be -1 for a non rootnode"
            assert self.current_node.children[action_index] == None, "action of the current node already has a node"
        else:
            root = True
            assert node.parent is None, "Root node should not have a parent"
        # Add node to tree
        node.tree = self
        self.nodes.append(node)
        # Add the node to the chain
        if not root:
            self.current_node.children[action_index] = node
            node.parent = self.current_node
            node.level = node.parent.level+1
            # Update all the nodes in the chain
            self.current_node.visit(node, node.values)
        if node.string_notation is not None:
            self.add_to_lookup_tables(node)
        # Check if we are done
        self.check_if_finished()
        # Start all over again
        self.current_node = self.nodes[0]
        self.descending = True
        self.current_action_index = -1

    def add_to_lookup_tables(self, node):
        self.values[node.string_notation] = node.values
        self.policies[node.string_notation] = node.policy

    def accumulated_policy(self):
        return self.root_node.accumulated_policy()
        

def visualize_mcts_tree(tree, fig, ax):
    graph = nx.Graph()

    # Sort the nodes
    def sort_nodes(node, sorted_nodes = []):
        sorted_nodes.append(node)
        for child in node.children:
            if child == None: continue
            sort_nodes(child, sorted_nodes)
        return sorted_nodes
    tree_nodes = sort_nodes(tree.root_node)

    edges = []
    nodes = []
    levels = [node.level for node in tree_nodes]
    max_level = np.max(levels)

    for index, node in enumerate(tree_nodes):
        node.index = index

    for node in tree_nodes:
        nodes.append({
            "index": node.index,
            "level": node.level,
            "width": len(node.children),
            "widths": [0 for _ in range(max_level+1)],
            "parent": None,
            "children": [],
            "annotation": node.annotation
        })
        for child, probability in zip(node.children, node.policy):
            if child == None:
                continue
            edges.append({
                "p":probability, 
                "from": node.index, 
                "to": child.index
            })
    
    # Link the nodes
    for edge in edges:
        parent = [node for node in nodes if node["index"]==edge["from"]][0]
        child = [node for node in nodes if node["index"]==edge["to"]][0]
        child["parent"] = parent
        parent["children"].append(child)
    
    def add_level_width(node, level, width):
        node["widths"][level]+=width
        if node["parent"] != None:
            add_level_width(node["parent"], level, width)
    # Get the max width down the tree
    for node in nodes:
        add_level_width(node, node["level"], node["width"])

    for level in range(max_level+1):
        x = 0
        for node in nodes:
            if node["level"] != level: continue
            graph.add_node(node["index"], pos=(x, node["level"]))
            x+=np.max(node["widths"])
    for edge in edges:
        graph.add_edge(edge["from"], edge["to"], width=edge["p"], capacity=edge["p"])
    pos=nx.get_node_attributes(graph,'pos')
    widths = list(nx.get_edge_attributes(graph,'width').values())
    widths = np.array(widths)
    if np.max(widths) == np.min(widths):
        widths = 3
    else:
        widths = 3 * (widths - np.min(widths)) / (np.max(widths) - np.min(widths))+1

    graph_nodes = nx.draw_networkx_nodes(graph, pos=pos, ax=ax)
    nx.draw_networkx_edges(graph, width=widths, pos=pos, ax=ax)
    # nodes = nx.draw_networkx(graph, with_labels=True, pos=pos, width=widths, ax=ax)
    annot = ax.annotate("", xy=(1,1), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

    # This shouldn't be needed but for some reason they thought it would be a good idea to mix up the positions
    graph_image_mapping_list = []
    mapping_index = 0
    for level in levels:
        for node in nodes:
            if node["level"] != level: continue
            graph_image_mapping_list.append(node["index"])
            mapping_index+=1
    
    annot.set_visible(False)
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = graph_nodes.contains(event)
            if cont:
                index = ind["ind"][0]
                mapped_index = graph_image_mapping_list[index]
                annot.xy = pos[mapped_index]
                node = nodes[mapped_index]
                annotation = node["annotation"]
                annot.set_text(f"{annotation}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)

if __name__ == "__main__":
    import pickle
    with open('temp.pickle', 'rb') as handle:
        tree = pickle.load(handle)
    visualize_mcts_tree(tree)