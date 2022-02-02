import numpy as np
import os
from connect_n import ConnectN
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, Conv2D, Embedding
from tensorflow.keras.models import Model

if __name__ == "__main__":

    # Size of the field
    size = (6, 7) # width and height
    number_of_players = 2
    n_descent = 100
    samples_per_epoch = 100
    number_of_train_samples_per_game = 5
    l2 = 1e-4

    policy_load_weights = "weights/connect_n_policy_network.h5"
    value_load_weights = "weights/connect_n_value_network.h5"

    number_of_actions = size[1]

    # Shared
    board_input = Input(size)
    player_input = Input((1,))

    # Create policy network
    b = Embedding(3,2)(board_input)
    # b = Conv2D(16, (4,4), 
    #     activation="relu",
    #     activity_regularizer=tf.keras.regularizers.L2(l2))(b)
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
    policy_network = Model((board_input, player_input), p)
    policy_network.compile(loss="binary_crossentropy", optimizer="adam")
    if policy_load_weights:
        policy_network.load_weights(policy_load_weights)

    # Create value network
    b = Embedding(3,2)(board_input)
    # b = Conv2D(16, (4,4), 
    #     activation="relu",
    #     activity_regularizer=tf.keras.regularizers.L2(l2))(b)
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

    game = ConnectN(size=size)

    step = 0
    while game.is_game_over == False:
        player = game.current_player-1
        board = game.board
        inputs = (board.reshape(1,*size), np.array([player]))
        actions = game.get_legal_actions()
        policy = policy_network(inputs)[0].numpy()
        values = value_network(inputs)[0].numpy()
        policy_actions = policy[actions]
        if step < 2:
            action = np.random.randint(number_of_players)
        elif step < 4:
            action = np.random.choice(actions, p=policy_actions)
        else:
            action = actions[np.argmax(policy_actions)]
        game.step(action)
        print(game)
        print(game.winner, values)
        print(np.array(100*policy,dtype=int))
        step+=1