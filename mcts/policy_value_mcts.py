import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

class PolicyValueNode():

    counter = 0

    def __init__(self, values, policy, actions, player, num_of_players, termination,
                string_notation:str=None, parent=None, annotation:str=None):
        
        assert len(policy) == len(actions), f"The length of the policy ({len(policy)}) must be the same a the number of actions ({len(actions)})"

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

        for action in self.actions:
            if isinstance(action,np.ndarray):
                # Atm I'm just to lazy to handle this case
                raise ValueError("Action may not be numpy arrays")

        self.number_of_edges = len(policy)
        
        self.qs = np.zeros((num_of_players, self.number_of_edges), dtype=float)
        self.ws = np.zeros((num_of_players, self.number_of_edges), dtype=float)
        self.ns = np.zeros(self.number_of_edges, dtype=float)

        self.parent = parent
        self.children = [None for _ in range(self.number_of_edges)]

        self.tree = None

    def u(self):
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
                
    

class PolicyValueTree():

    def __init__(self, root_node, 
        cpuct=0.3, dirichelt_epsilon=0.25, n_descent=100,
        shared_values=None, shared_policies=None):

        self.nodes = []

        self.set_tree_to_node(root_node)

        self.cpuct = cpuct
        self.dirichelt_epsilon = dirichelt_epsilon
        self.n_descent = n_descent

        self.root_node = root_node
        self.current_node = self.root_node

        # Holds the values belonging to string notation
        self.values = {} if shared_values == None else shared_values
        self.policies = {} if shared_policies == None else shared_policies
        if root_node.string_notation != None:
            self.values[self.root_node.string_notation] = self.root_node.values
            self.policies[self.root_node.string_notation] = self.root_node.policy

        self.is_finished = False
        self.descending = True

    def get_string_notation(self):
        return self.current_node.string_notation

    def set_tree_to_node(self, node):
        assert node.tree == None, "Tree is already set to the root node"
        # Add to all the nodes
        self.nodes.append(node)
        node.tree = self

    def get_action(self):
        node = self.current_node
        index = np.argmax(node.q() + node.u())
        return (node.actions[index]), index, node.children[index]

    def step(self):
        action, action_index, node = self.get_action()
        # Check if it's a leave node
        if node == None:
            self.descending = False
            return action, action_index
        elif node.termination:
            # Handle termination nodes
            self.current_node.visit(node, node.values)
            self.current_node = self.root_node
        else:
            # Set the next node as the current node
            self.current_node = node
        self.check_if_finished()

    def check_if_finished(self):
        self.is_finished = self.root_node.n() >= self.n_descent
        if self.is_finished:
            self.descending = False
        return self.is_finished

    def get_policy_and_value(self, string_notation):
        if string_notation in self.values and string_notation in self.policies:
            return self.policies[string_notation], self.values[string_notation]
        return None, None

    def add_node(self, action_index, node):
        assert self.descending == False, "Tree is still descending"
        assert self.current_node.children[action_index] == None, "action of the current node already has a node"
        # Set the tree to the node
        self.set_tree_to_node(node)
        # Add the node to the chain
        self.current_node.children[action_index] = node
        node.parent = self.current_node
        node.level = node.parent.level+1
        # Update all the nodes in the chain
        self.current_node.visit(node, node.values)
        if node.string_notation != None:
            self.values[node.string_notation] = node.values
            self.policies[node.string_notation] = node.policy
        # Check if we are done
        self.check_if_finished()
        # Start all over again
        self.current_node = self.root_node
        self.descending = True

    def accumulated_policy(self):
        return self.root_node.accumulated_policy()
        

def add_childeren(node):
    pass

def visualize_mcts_tree(tree):
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
    widths = 3 * (widths - np.min(widths)) / (np.max(widths) - np.min(widths))+1



    fig, ax = plt.subplots()
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

    plt.show()

if __name__ == "__main__":
    import pickle
    with open('temp.pickle', 'rb') as handle:
        tree = pickle.load(handle)
    visualize_mcts_tree(tree)