import numpy as np
import os
from connect_n import ConnectN
from mcts.nodes import PolicyValueNode
from mcts.trees import PolicyValueTree, visualize_mcts_tree
import tensorflow as tf
import torch
from torch import nn
# import multiprocessing as mp
from torch import multiprocessing as mp
import torch.nn.functional as F
import time
import pickle
from scipy.special import softmax

class MCTSSessionProcess(mp.Process):

    def __init__(self, n_descent, number_of_players=2, 
        init_temperature=1, min_temperature=0, steps_temperature=10, 
        cpuct=0.1,
        shared_values=None, shared_policies=None,
        **kwargs):
        super(MCTSSessionProcess, self).__init__(**kwargs)
        self.seed = np.random.randint(int(1e7))
        self.n_descent = n_descent
        self.number_of_players = number_of_players
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.results_queue = mp.Queue()

        self.min_temperature = min_temperature
        self.init_temperature = init_temperature
        self.steps_temperature = steps_temperature
        self.temperature = self.init_temperature

        self.cpuct = cpuct
        self.shared_values = shared_values
        self.shared_policies = shared_policies

    def _temperature_step(self):
        self.temperature -= (self.init_temperature - self.min_temperature) / self.steps_temperature
        if self.temperature <= 0 + 1e-8:
            self.temperature = 0

    def run(self):
        np.random.seed(self.seed)

        # Start a new game
        game = ConnectN(size=size)

        number_of_actions = len(game.get_legal_actions())

        game_history = {
            "accumulated_policies": [],
            "current_players": [],
            "boards": [],
            "winner": -1
        }

        # Run game until the game has ended
        while game.is_game_over == False:

            # Get the first state
            root_board = game.board
            root_player = game.current_player - 1
            # root_current_player = np.zeros((1, self.number_of_players))
            # root_current_player[0, player-1] = 1
            string_notation = game.string_notation()
            termination = game.is_game_over

            # Predict the policy and value
            root_inputs = (root_board, root_player)
            self.request_queue.put(root_inputs)
            policy, values = self.response_queue.get()
            root_actions = game.get_legal_actions()
            root_policy = policy.take(root_actions)

            # Create the root node
            root_node = PolicyValueNode(values, root_policy, root_actions, root_player, self.number_of_players, termination,
                string_notation=string_notation, annotation=str(game))

            # Create tree
            tree = PolicyValueTree(root_node, 
                n_descent=self.n_descent, cpuct=self.cpuct,
                shared_policies=self.shared_policies, shared_values=self.shared_values
            )

            # Run the mcts process until the tree is done
            while tree.is_finished == False:

                # Decent the tree until a leave node
                while tree.descending:
                    out = tree.step()

                if tree.is_finished:
                    break

                action, action_index = out

                # Play the action of the leave node
                parent_string_notation = tree.get_string_notation()

                # Create the game state of the new node
                mcts_game = ConnectN(string_notation=parent_string_notation, size=size)
                mcts_game.step(action)

                # Get the information of this state
                player = mcts_game.current_player - 1
                board = mcts_game.board
                termination = mcts_game.is_game_over
                child_string_notation = mcts_game.string_notation()
                # current_player = np.zeros(self.number_of_players)
                # current_player[player-1] = 1
                actions = mcts_game.get_legal_actions()

                # The input for the network
                inputs = (board, player)

                # Get the policy and values
                policy, values = tree.get_policy_and_value(child_string_notation) # Check if the state is already known to the tree
                if policy is None:
                    # Queue input
                    self.request_queue.put(inputs)
                    raw_policy, raw_values = self.response_queue.get()

                    policy = raw_policy.take(actions)
                    if termination:
                        values = -np.ones(self.number_of_players)
                        winner = mcts_game.winner
                        values[winner-1] = 1
                        # print(mcts_game)
                        # print(values)
                        # print(actions)
                        # print(policy)
                    else:
                        values = raw_values

                # Create a new node
                node = PolicyValueNode(values, policy, actions, player, self.number_of_players, termination,
                    string_notation=child_string_notation, annotation=str(mcts_game))
                    
                # Add node to the tree
                tree.add_node(action_index, node)

            # print('a')
            # visualize_mcts_tree(tree)
            # with open('temp.pickle', 'wb') as handle:
            #     pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Gather the results of MCTS
            accumulated_policy = tree.accumulated_policy()
            train_accumulated_policy = np.zeros(number_of_actions)
            train_accumulated_policy[root_actions] = accumulated_policy

            # Add results to game history
            game_history["accumulated_policies"].append(train_accumulated_policy)
            game_history["boards"].append(root_inputs[0])
            game_history["current_players"].append(root_inputs[1])

            if self.temperature == 0:
                # Take best action
                action = root_actions[np.argmax(accumulated_policy)]
            else:
                # Draw an action from the policy
                hot_accumulated_policy = accumulated_policy ** (1/self.temperature) / np.sum(accumulated_policy ** (1/self.temperature))
                action = np.random.choice(root_actions, p=hot_accumulated_policy)
                self._temperature_step()

            # print(root_board)
            # print(root_current_player)
            # print(root_actions)
            # print(root_policy)
            # print(accumulated_policy)
            # print(root_node.qs)
            # print(action)
            
            # Peform a step
            game.step(action)

        # Add winner to game history
        game_history["winner"] = game.winner

        self.results_queue.put(game_history)


class PolicyNetwork(nn.Module):

    def __init__(self, board_size=(6,7), num_players=2, enc_dim=2, linear_layers_dims=[64,32,16], **kwargs):
        super(PolicyNetwork, self).__init__()
        self.player_dim = num_players + 1
        self.enc_dim = enc_dim
        self.board_size = board_size
        self.board_len = np.prod(board_size)

        self.conv_filters = 16

        self.player_emb = nn.Embedding(self.player_dim, self.enc_dim)

        self.conv = nn.Conv2d(self.enc_dim, self.conv_filters, (4, 4))

        dim = (self.board_size[0] - 3) * (self.board_size[1] -3) * self.conv_filters + self.enc_dim
        
        self.linear_layers = []
        for layer_dim in linear_layers_dims:
            linear_layer = torch.nn.Linear(dim, layer_dim)
            self.linear_layers.append(linear_layer)
            dim = layer_dim
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.output_layer = torch.nn.Linear(dim, self.board_size[1])

    def forward(self, boards, turn_players):
        batch_size = boards.shape[0]

        boards = self.player_emb(boards)
        boards = F.relu(self.conv(boards.permute(0, 3, 1, 2)))
        boards = boards.reshape(batch_size, -1)

        turn_players = turn_players.view(batch_size, 1)
        turn_players = self.player_emb(turn_players)
        turn_players = turn_players.view(batch_size, -1)

        x = torch.concat((boards, turn_players),axis=-1)
        
        for linear_layer in self.linear_layers:
            x = F.relu(linear_layer(x))

        return self.output_layer(x)


class ValueNetwork(nn.Module):

    def __init__(self, board_size=(6,7), num_players=2, enc_dim=2, linear_layers_dims=[64,32,16], **kwargs):
        super(ValueNetwork, self).__init__()
        self.player_dim = num_players + 1
        self.enc_dim = enc_dim
        self.board_size = board_size
        self.board_len = np.prod(board_size)

        self.conv_filters = 16

        self.player_emb = nn.Embedding(self.player_dim, self.enc_dim)

        self.conv = nn.Conv2d(self.enc_dim, self.conv_filters, (4, 4))

        dim = (self.board_size[0] - 3) * (self.board_size[1] -3) * self.conv_filters + self.enc_dim
        self.linear_layers = []
        for layer_dim in linear_layers_dims:
            linear_layer = torch.nn.Linear(dim, layer_dim)
            self.linear_layers.append(linear_layer)
            dim = layer_dim
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.output_layer = torch.nn.Linear(dim, num_players)

    def forward(self, boards, turn_players):
        batch_size = boards.shape[0]

        boards = self.player_emb(boards)
        boards = F.relu(self.conv(boards.permute(0, 3, 1, 2)))
        boards = boards.reshape(batch_size, -1)

        turn_players = turn_players.view(batch_size, 1)
        turn_players = self.player_emb(turn_players)
        turn_players = turn_players.view(batch_size, -1)

        x = torch.concat((boards, turn_players),axis=-1)

        for linear_layer in self.linear_layers:
            x = F.relu(linear_layer(x))

        return self.output_layer(x)


def policy_networks_play(policy_networks, number_of_games=100):
    winners = np.zeros(number_of_games, dtype=int)
    for game_index in range(number_of_games):
        game = ConnectN(size=size)
        step = 0
        while game.is_game_over == False:
            player = game.current_player
            policy_network = policy_networks[player-1]
            board = game.board
            inputs = (board.reshape((1,*size)), np.array([current_player]))
            actions = game.get_legal_actions()
            policy = policy_network.predict(inputs)[0]
            policy_actions = policy[actions]
            if step < 2:
                action = np.random.randint(number_of_players)
            elif step < 4:
                action = np.random.choice(actions, p=policy_actions)
            else:
                action = actions[np.argmax(policy_actions)]
            game.step(action)
            step+=1
        winners[game_index] = game.winner
    return winners

def value_networks_play(value_networks, number_of_games=100):
    winners = np.zeros(number_of_games, dtype=int)
    for game_index in range(number_of_games):
        game = ConnectN(size=size)
        step = 0
        while game.is_game_over == False:
            turn_player = game.current_player -1
            value_network = value_networks[turn_player]
            legal_actions = game.get_legal_actions()
            number_of_legal_actions = len(legal_actions)
            boards = np.zeros((number_of_legal_actions, *game.size))
            current_players = np.zeros((number_of_legal_actions, 1), dtype=int)
            for legal_action_index, legal_action in enumerate(legal_actions):
                action_game = ConnectN(string_notation=game.string_notation())
                action_game.step(legal_action)
                boards[legal_action_index] = action_game.board
                player = action_game.current_player -1
                current_players[legal_action_index, 0] = player
            values = value_network.predict((boards, current_players))
            legal_action_values = values[:, turn_player]
            legal_action_policy = softmax(legal_action_values)

            if step < 2:
                action = np.random.randint(number_of_players)
            elif step < 4:
                action = np.random.choice(legal_actions, p=legal_action_policy)
            else:
                action = legal_actions[np.argmax(legal_action_policy)]
            game.step(action)
            # print(game)
            # print(legal_actions)
            # print(values)
            # print(action)
            step+=1
        winners[game_index] = game.winner
    return winners

if __name__ == "__main__":

    size = (6, 7)                           # height and width of board
    number_of_players = 2                   # The number of players
    n_descent = 100                         # How many path are taking during a MCTS
    number_of_epochs = 1#_000_000                  # The number of epochs
    number_of_games_per_epoch = 10         # The minimal number of samples per epoch
    number_of_train_samples_per_game = 5    # The number of random training samples per game
    number_of_processes = 16                # Number of parallel processes
    batch_size = 8                          # The number of samples holding during running model
    cpuct = 0.1
    l2 = 1e-4
    init_temperature = 6

    batch_size = np.min((batch_size, number_of_processes))

    policy_load_weights = "weights/connect_n_policy_network.h5"
    value_load_weights = "weights/connect_n_value_network.h5"

    policy_save_weights = "weights/connect_n_policy_network.h5"
    value_save_weights = "weights/connect_n_value_network.h5"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    policy_network = PolicyNetwork()
    if policy_load_weights != None:
        policy_network.load_state_dict(torch.load(policy_load_weights))
    policy_network.to(device)
    policy_loss = torch.nn.CrossEntropyLoss()
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-4)

    value_network = ValueNetwork()
    if value_load_weights != None:
        value_network.load_state_dict(torch.load(value_load_weights))
    value_network.to(device)
    value_loss = torch.nn.MSELoss()
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=1e-4)

    manager = mp.Manager()

    shared_policies = manager.dict()
    shared_values = manager.dict()

    # Create processes
    mcts_processes = [MCTSSessionProcess(
        n_descent=n_descent, cpuct=cpuct, init_temperature=init_temperature,
        shared_policies=shared_policies, shared_values=shared_values) for index in range(number_of_processes)]
    # Start processes
    for mcts_process in mcts_processes: 
        mcts_process.start()

    waiting_mcts = {}

    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch}")

        shared_policies.clear()
        shared_values.clear()

        # Initiate trainings data lists
        train_boards = []
        train_current_players = []
        train_values = []
        train_policies = []

        # Handling the MCTS processes requests
        start_time = time.time()
        game_histories = []
        while len(game_histories) < number_of_games_per_epoch:
            # print(len(shared_policies))
            for mcts_process in mcts_processes:
                if mcts_process.request_queue.qsize() > 0:
                    waiting_mcts[mcts_process] = mcts_process.request_queue.get()
                elif mcts_process.results_queue.qsize() > 0:
                    game_history = mcts_process.results_queue.get()
                    game_histories.append(game_history)
                    mcts_processes.remove(mcts_process)
                    new_mcts_process = MCTSSessionProcess(
                        n_descent=n_descent, cpuct=cpuct, init_temperature=init_temperature,
                        shared_policies=shared_policies, shared_values=shared_values)
                    new_mcts_process.start()
                    mcts_processes.append(new_mcts_process)

                current_batch_size = len(waiting_mcts)
                if current_batch_size >= batch_size:
                    boards = np.zeros((current_batch_size, *size))
                    players = np.zeros((current_batch_size, 1))
                    mctses = []
                    mctses = [key for key in waiting_mcts.keys()]
                    inputs = [value for value in waiting_mcts.values()]
                    for index, (mcts_process, (board, player)) in enumerate(zip(mctses, inputs)):
                        boards[index] = board
                        players[index] = player
                        mctses.append(mcts_process)
                        del waiting_mcts[mcts_process] # Remove the mcts from waiting list
                    inputs = (torch.tensor(boards, dtype=int).to(device), torch.tensor(players, dtype=int).to(device))
                    # Run the inputs through the network
                    with torch.no_grad():
                        out = policy_network(*inputs)
                    policies = torch.softmax(out, dim=-1).cpu().numpy()
                    # print(b)
                    with torch.no_grad():
                        values = value_network(*inputs)
                    values = values.cpu().numpy()
                    # Return the responses to the processes
                    for mcts_process, policy, value in zip(mctses, policies, values):
                        mcts_process.response_queue.put((policy, value))

        # print(game_histories[0])
        for game_history in game_histories:
            # Create the target values
            if game_history["winner"] == -1:
                zs = np.zeros(number_of_players)
            else:
                zs = -np.ones(number_of_players)
                zs[game_history["winner"]-1] = 1

            # Take a sample of the game
            number_of_steps = len(game_history["boards"])
            num_train_samples = np.min((number_of_train_samples_per_game, number_of_steps))
            train_indices = np.random.choice(np.arange(number_of_steps), num_train_samples)

            # for i in [-4,-3,-2,-1]:
            #     print(i)
            #     print(game_history["boards"][i])
            #     print(game_history["accumulated_policies"][i])
            #     print(game_history["current_players"][i])
            # print(zs)

            sample_train_accumulated_policies = [game_history["accumulated_policies"][index] for index in train_indices]
            sample_train_boards = [game_history["boards"][index] for index in train_indices]
            sample_train_current_players = [game_history["current_players"][index] for index in train_indices]

            # Add the samples to the trainings data
            for index, (accumulated_policy, board, current_player) in enumerate(zip(sample_train_accumulated_policies, 
                                                                            sample_train_boards, 
                                                                            sample_train_current_players)):
                # print("b", board)
                # print("p", current_player)
                train_boards.append(board)
                train_current_players.append([current_player])
                train_policies.append(accumulated_policy)
                train_values.append(zs)

        delta_time = time.time() - start_time
        history_length = len(game_histories)
        print(f"MCTS duration: {delta_time} for {history_length} games, {delta_time/history_length} s/game")
 

        # Train
        print("Train")
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_boards, dtype=int), 
            torch.tensor(train_current_players, dtype=int),
            torch.tensor(np.array(train_policies), dtype=torch.float32),
            torch.tensor(np.array(train_values), dtype=torch.float32))

        loader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                    pin_memory=True)
                    
        # Policy network
        current_policy_network = PolicyNetwork()
        current_policy_network.load_state_dict(policy_network.state_dict())

        # Value network
        # current_value_network = value_network
        current_value_network = ValueNetwork()
        current_value_network.load_state_dict(value_network.state_dict())

        for boards, train_current_players, train_policies, train_values in loader:

            inputs = (boards.to(device), train_current_players.to(device))
            train_policies = train_policies.to(device)
            train_values = train_values.to(device)

            # Train the network
            # for i in range(len(train_values)):
            #     print(inputs[0][i])
            #     print(inputs[1][i])
            #     print(train_policies[i])
            policy_optimizer.zero_grad()
            policy_logits = policy_network(*inputs)
            policy_losses = policy_loss(policy_logits, train_policies)
            policy_losses.backward()
            policy_optimizer.step()

            for p, t in zip(torch.softmax(policy_logits[:10], dim=-1), train_policies[:10]):
                print("p", p)
                print("t", t)

            # Train the network
            # for i in range(len(train_values)):
            #     print(inputs[0][i])
            #     print(inputs[1][i])
            #     print(train_values[i])

            value_optimizer.zero_grad()
            value_logits = value_network(*inputs)
            value_losses = value_loss(value_logits, train_values)
            value_losses.backward()
            value_optimizer.step()

            # for p, t in zip(value_logits[:10], train_values[:10]):
            #     print("p", p)
            #     print("t", t)

        # Testing model against it's previous self
        # policy_play_results = policy_networks_play(policy_networks=(policy_network, current_policy_network))
        policy_play_results = 1
        new_policy_win_rate = np.sum(policy_play_results == 1) / np.sum(policy_play_results != -1)
        print("policy", policy_play_results, new_policy_win_rate)

        # Save or keep
        if new_policy_win_rate < 0.45:
            print("Keep policy weights")
            policy_network.load_state_dict(current_policy_network.state_dict())
        else:
            # Save the weights
            print("Save policy network")
            torch.save(policy_network.state_dict(), policy_save_weights)

        # Testing model against it's previous self
        # value_play_results = value_networks_play((value_network, value_network))
        value_play_results = 1
        new_value_win_rate = np.sum(value_play_results == 1) / np.sum(value_play_results != -1)
        print("value", value_play_results, new_value_win_rate)

        # Save or keep
        if new_value_win_rate < 0.45:
            print("Keep value weights")
            value_network.load_state_dict(current_value_network.state_dict())
        else:
            # Save the weights
            print("Save value network")
            torch.save(value_network.state_dict(), value_save_weights)

    print("Done")
    for mcts_process in mcts_processes:
        mcts_process.terminate()
