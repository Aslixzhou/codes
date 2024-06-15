# import the necessary libraries
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

class HMM:
    def __init__(self, num_states, num_obs):
        self.num_states = num_states
        self.num_obs = num_obs
        self.transition_matrix = np.zeros((num_states, num_states))
        self.emission_matrix = np.zeros((num_states, num_obs))
        self.initial_state_probs = np.zeros(num_states)

    def train_supervised(self, sequences, state_sequences):
        # 估计初始状态概率
        for state_seq in state_sequences:
            self.initial_state_probs[state_seq[0]] += 1
        self.initial_state_probs /= len(state_sequences)

        # 估计状态转移概率矩阵
        for state_seq in state_sequences:
            for i in range(len(state_seq) - 1):
                self.transition_matrix[state_seq[i], state_seq[i+1]] += 1
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

        # 估计观测概率矩阵
        for i in range(len(sequences)):
            obs_seq = sequences[i]
            state_seq = state_sequences[i]
            for j in range(len(obs_seq)):
                self.emission_matrix[state_seq[j], obs_seq[j]] += 1
        row_sums = self.emission_matrix.sum(axis=1)
        self.emission_matrix = self.emission_matrix / row_sums[:, np.newaxis]

    def generate_sequence(self, length):
        # 生成序列的代码
        pass

if __name__ == '__main__':
    # Define the state space
    states = ["Sunny", "Rainy"] # 隐藏序列集 晴天还是雨天
    n_states = len(states)
    print('Number of hidden states :', n_states)
    # Define the observation space
    observations = ["Dry", "Wet"] # 观测序列集 干还是湿
    n_observations = len(observations)
    print('Number of observations :', n_observations)

    # Define the initial state distribution
    state_probability = np.array([0.6, 0.4]) # 初始状态概率分布 初始晴天还是雨天的概率分布
    print("State probability: ", state_probability)
    # Define the state transition probabilities
    transition_probability = np.array([[0.7, 0.3],
                                       [0.3, 0.7]]) # 初始状态概率分布 晴天雨天转移概率分布 晴天->晴天 晴天->雨天 雨天->晴天 雨天->雨天
    print("\nTransition probability:\n", transition_probability)
    # Define the observation likelihoods
    emission_probability = np.array([[0.9, 0.1],
                                     [0.2, 0.8]]) # 观测状态概率分布 晴天下：干0.9 湿0.1 雨天下：干0.2 湿0.8
    print("\nEmission probability:\n", emission_probability)

    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = state_probability # 初始状态概率分布
    model.transmat_ = transition_probability # 初始状态概率分布
    model.emissionprob_ = emission_probability # 观测状态概率分布

    '''预测隐藏序列问题'''

    # 观测序列（干湿） 预测 隐藏序列（晴天雨天）
    # Define the sequence of observations
    observations_sequence = np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1)
    print(observations_sequence)
    # 根据观测序列预测隐藏状态序列 晴天还是雨天
    # Predict the most likely sequence of hidden states
    hidden_states = model.predict(observations_sequence)
    print("Most likely hidden states:", hidden_states)

    '''求解观测序列概率问题'''

    # 求解观测序列的概率问题
    print("观测序列概率：",math.exp(model.score(observations_sequence)))

    # 未来天气预测
    # 定义状态空间
    states = ['晴天', '雨天']
    # 定义转移矩阵 (transition matrix)
    # 该矩阵表示从当前状态到下一个状态的转移概率
    transition_matrix = np.array([
        [0.8, 0.2],  # 从晴天到晴天的概率为 0.8，从晴天到雨天的概率为 0.2
        [0.4, 0.6]  # 从雨天到晴天的概率为 0.4，从雨天到雨天的概率为 0.6
    ])

    # 定义初始状态分布 (initial state distribution)
    # 这里假设初始时刻晴天和雨天的概率分别为 0.5
    initial_distribution = np.array([0.5, 0.5])

    '''未来预测问题'''

    # 预测未来 5 天的天气情况
    num_days = 5
    current_state = np.random.choice(states, p=initial_distribution) # 当前状态
    predicted_weather = [current_state]

    for _ in range(num_days):
        # 根据当前状态进行状态转移
        next_state = np.random.choice(states, p=transition_matrix[states.index(current_state)])
        predicted_weather.append(next_state)
        current_state = next_state
    print("未来 5 天的天气预测为:", predicted_weather)


    '''参数估计'''
    print("------------隐马尔可夫模型参数估计-------------")
    sequences = [[0, 1, 0, 2, 1], [1, 2, 0, 1, 2]]
    state_sequences = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
    hmm_model = HMM(num_states=2, num_obs=3)
    hmm_model.train_supervised(sequences, state_sequences)
    print("Initial state probabilities:", hmm_model.initial_state_probs)
    print("Transition matrix:\n", hmm_model.transition_matrix)
    print("Emission matrix:\n", hmm_model.emission_matrix)
