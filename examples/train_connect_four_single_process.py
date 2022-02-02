import numpy as np
import os
from connect_n import ConnectN
from mcts import PolicyValueNode, PolicyValueTree
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model


if __name__ == "__main__":

    size = (6, 7)                           # height and width of board
    number_of_players = 2                   # The number of players
    n_descent = 100                         # How many path are taking during a MCTS
    number_of_epochs = 100                  # The number of epochs
    samples_per_epoch = 100                 # The minimal number of samples per epoch
    number_of_train_samples_per_game = 5    # The number of random training samples per game

    policy_load_weights = "weights/connect_n_policy_network.h5"
    value_load_weights = "weights/connect_n_value_network.h5"

    policy_save_weights = "weights/connect_n_policy_network.h5"
    value_save_weights = "weights/connect_n_value_network.h5"

    number_of_actions = size[1]

    # Inputs
    board_input = Input(size)
    player_input = Input((number_of_players,))
    combined_input = concatenate((Flatten()(board_input), player_input))

    # Create policy network
    p = Dense(32, activation="relu")(combined_input)
    p = Dense(16, activation="relu")(p)
    p = Dense(number_of_actions, activation="softmax")(p)
    policy_network = Model((board_input, player_input), p)
    policy_network.compile(loss="binary_crossentropy", optimizer="adam")
    if policy_load_weights:
        policy_network.load_weights(policy_load_weights)

    # Create value network
    v = Dense(32, activation="relu")(combined_input)
    v = Dense(16, activation="relu")(v)
    v = Dense(number_of_players, activation="sigmoid")(v)
    value_network = Model((board_input, player_input), v)
    value_network.compile(loss="mse", optimizer="adam")
    if value_load_weights:
        value_network.load_weights(value_load_weights)

    for epoch in range(number_of_epochs):

        # Initiate trainings data lists
        train_boards = []
        train_current_players = []
        train_values = []
        train_policies = []

        while len(train_boards) < samples_per_epoch:

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
                board = game.board
                player = game.current_player
                string_notation = game.string_notation()
                termination = game.is_game_over

                # Predict the policy and value
                inputs = (board.reshape(1,*size), np.array([[1, 0]]))
                policy = policy_network(inputs)[0].numpy()
                values = value_network(inputs)[0].numpy()
                root_actions = game.get_legal_actions()

                # Create the root node
                root_node = PolicyValueNode(values, policy.take(root_actions), root_actions, player-1, number_of_players, termination,
                    string_notation=string_notation)

                # Create tree
                tree = PolicyValueTree(root_node, 
                    n_descent=n_descent)

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
                    player = mcts_game.current_player
                    board = mcts_game.board
                    termination = mcts_game.is_game_over
                    child_string_notation = mcts_game.string_notation()
                    current_player = np.zeros((1,number_of_players))
                    current_player[0,player-1] = 1
                    actions = mcts_game.get_legal_actions()

                    # The input for the network
                    inputs = (board.reshape(1,*size), current_player)

                    # Get the policy and values
                    policy, values = tree.get_policy_and_value(child_string_notation) # Check if the state is already known to the tree
                    if policy is None:
                        policy = policy_network(inputs)[0].numpy().take(actions)
                    if values is None:
                        if termination:
                            values = - np.ones(number_of_players)
                            winner = mcts_game.winner
                            values[winner-1] = 1
                        else:
                            values = value_network(inputs)[0].numpy()

                    # Create a new node
                    node = PolicyValueNode(values, policy, actions, player-1, number_of_players, termination,
                        string_notation=child_string_notation)

                    # Add node to the tree
                    tree.add_node(action_index, node)

                # Gather the results of MCTS
                accumulated_policy = tree.accumulated_policy()
                action = root_actions[np.argmax(accumulated_policy)]
                train_accumulated_policy = np.zeros(number_of_actions)
                train_accumulated_policy[root_actions] = accumulated_policy

                # Add results to game history
                game_history["accumulated_policies"].append(train_accumulated_policy)
                game_history["boards"].append(inputs[0][0])
                game_history["current_players"].append(inputs[1][0])

                # Peform a step
                game.step(action)

            # Add winner to game history
            game_history["winner"] = game.winner

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

            sample_train_accumulated_policies = [game_history["accumulated_policies"][index] for index in train_indices]
            sample_train_boards = [game_history["boards"][index] for index in train_indices]
            sample_train_current_players = [game_history["current_players"][index] for index in train_indices]

            # Add the samples to the trainings data
            for index, (accumulated_policy, board, current_player) in enumerate(zip(sample_train_accumulated_policies, 
                                                                            sample_train_boards, 
                                                                            sample_train_current_players)):
                train_boards.append(board)
                train_current_players.append(current_player)
                train_policies.append(accumulated_policy)
                train_values.append(zs)

        # Train
        inputs = (np.array(train_boards), np.array(train_current_players))
        policy = policy_network.fit(inputs, np.array(train_policies))
        values = value_network.fit(inputs, np.array(train_values))

        # Save the weights
        policy_network.save_weights(policy_save_weights)
        value_network.save_weights(value_save_weights)

    print("Done")