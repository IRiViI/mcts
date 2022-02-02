import numpy as np
import os
from connect_n import ConnectN
from mcts import PolicyValueNode, PolicyValueTree, visualize_mcts_tree
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, Conv2D, Embedding
from tensorflow.keras.models import Model
import multiprocessing as mp
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
        self.seed = np.random.randint(1e7)
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

    # tf.compat.v1.disable_eager_execution()

    size = (6, 7)                           # height and width of board
    number_of_players = 2                   # The number of players
    n_descent = 100                         # How many path are taking during a MCTS
    number_of_epochs = 1                  # The number of epochs
    number_of_games_per_epoch = 10         # The minimal number of samples per epoch
    number_of_train_samples_per_game = 5    # The number of random training samples per game
    number_of_processes = 16                # Number of parallel processes
    batch_size = 8                          # The number of samples holding during running model
    cpuct = 0.1
    l2 = 1e-4
    init_temperature = 6

    physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     pass

    batch_size = np.min((batch_size, number_of_processes))

    policy_load_weights = None # "weights/connect_n_policy_network.h5"
    value_load_weights = None #"weights/connect_n_value_network.h5"

    policy_save_weights = "weights/connect_n_policy_network.h5"
    value_save_weights = "weights/connect_n_value_network.h5"

    number_of_actions = size[1]

    # Shared
    board_input = Input(size)
    player_input = Input((1,))

    # Create policy network
    b = Embedding(3,2)(board_input)
    b = Conv2D(16, (5,5),
        padding="same", 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(b)
    b = Flatten()(b)
    b = concatenate((b, Flatten()(Embedding(2,2)(player_input))))
    p = Dense(64, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(b)
    p = Dense(32, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(p)
    p = Dense(16, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(p)
    p = Dense(number_of_actions, activation="softmax")(p)
    policy_network = Model((board_input, player_input), p )
    policy_network.compile(loss="binary_crossentropy", optimizer="adam")
    if policy_load_weights:
        policy_network.load_weights(policy_load_weights)

    # Create value network
    b = Embedding(3,2)(board_input)
    b = Conv2D(16, (5,5),
        padding="same", 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(b)
    b = Flatten()(b)
    b = concatenate((b, Flatten()(Embedding(2,2)(player_input))))
    v = Dense(64, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(b)
    v = Dense(32, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(v)
    v = Dense(16, 
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L2(l2))(v)
    v = Dense(number_of_players)(v)
    value_network = Model((board_input, player_input), v)
    value_network.compile(loss="mse", optimizer="adam")
    if value_load_weights:
        value_network.load_weights(value_load_weights)

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
                    inputs = (boards, players)
                    # Run the inputs through the network
                    policies = policy_network.predict(inputs)
                    # print(b)
                    values = value_network.predict(inputs)
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
        inputs = (np.array(train_boards), np.array(train_current_players))

        # Policy network
        current_policy_network = tf.keras.models.clone_model(policy_network)

        # Train the network
        # for i in range(len(train_values)):
        #     print(inputs[0][i])
        #     print(inputs[1][i])
        #     print(train_policies[i])
        policy_network.fit(inputs, np.array(train_policies))

        # Testing model against it's previous self
        # policy_play_results = policy_networks_play(policy_networks=(policy_network, current_policy_network))
        policy_play_results = 1
        new_policy_win_rate = np.sum(policy_play_results == 1) / np.sum(policy_play_results != -1)
        print("policy", policy_play_results, new_policy_win_rate)

        # Save or keep
        if new_policy_win_rate < 0.45:
            print("Keep policy weights")
            policy_network.set_weights(current_policy_network.get_weights())
        else:
            # Save the weights
            print("Save policy network")
            policy_network.save_weights(policy_save_weights)

        # Value network
        current_value_network = tf.keras.models.clone_model(value_network)

        # Train the network
        for i in range(len(train_values)):
            print(inputs[0][i])
            print(inputs[1][i])
            print(train_values[i])
        value_network.fit(inputs, np.array(train_values))

        # Testing model against it's previous self
        # value_play_results = value_networks_play((value_network, value_network))
        value_play_results = 1
        new_value_win_rate = np.sum(value_play_results == 1) / np.sum(value_play_results != -1)
        print("value", value_play_results, new_value_win_rate)

        # Save or keep
        if new_value_win_rate < 0.45:
            print("Keep value weights")
            value_network.set_weights(current_value_network.get_weights())
        else:
            # Save the weights
            print("Save value network")
            value_network.save_weights(value_save_weights)

    print("Done")
    for mcts_process in mcts_processes:
        mcts_process.terminate()
