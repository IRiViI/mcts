import numpy as np
import os
from connect_n import ConnectN
import tensorflow as tf
import torch
from train_connect_four_multiple_processes import PolicyNetwork, ValueNetwork

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    policy_network = PolicyNetwork()
    if policy_load_weights != None:
        policy_network.load_state_dict(torch.load(policy_load_weights))
    policy_network.to(device)

    value_network = ValueNetwork()
    if value_load_weights != None:
        value_network.load_state_dict(torch.load(value_load_weights))
    value_network.to(device)

    game = ConnectN(size=size)

    step = 0
    while game.is_game_over == False:
        player = game.current_player
        board = game.board
        inputs = (
            torch.tensor(board, dtype=int).view(1,*size).to(device), 
            torch.tensor([player],dtype=int).to(device)
        )
        actions = game.get_legal_actions()
        with torch.no_grad():
            policies = torch.softmax(policy_network(*inputs), dim=-1)
        policies = policies.cpu().numpy()[0]
        with torch.no_grad():
            values = value_network(*inputs)
        values = values.cpu().numpy()[0]
        policy_actions = policies[actions]
        if step < 2:
            action = np.random.randint(number_of_actions)
        elif step < 4:
            action = np.random.choice(actions, p=policy_actions)
        else:
            action = actions[np.argmax(policy_actions)]
        print(game)
        print(actions)
        print(policy_actions)
        print(action)
        game.step(action)
        print(policies)
        print(game.winner, values)
        # print(100*policies)
        step+=1