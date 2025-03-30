import numpy as np


def model_1(obs):
    deviate_1 = int(np.random.choice([0, 1]))
    deviate_2 = int(np.random.choice([0, 1]))
    return [deviate_1,deviate_2]

def model_2(obs):
    prob = [0.4,0.4]
    deviate_1 = np.random.choice([0, 1], p=[1-prob[0], prob[0]])
    deviate_2 = np.random.choice([0, 1], p=[1-prob[1], prob[1]])
    return [deviate_1,deviate_2]
def model_3(obs):
    prob = []
    for i in [1,3]:
        if obs[i] >= obs[i+1]:
            prob.append(0.6)
        else:
            prob.append(0.2)
    deviate_1 = np.random.choice([0, 1], p=[1-prob[0], prob[0]])
    deviate_2 = np.random.choice([0, 1], p=[1-prob[1], prob[1]])
    direct_a = prob
    control_a = [deviate_1,deviate_2]
    return direct_a
    