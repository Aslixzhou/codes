import numpy as np

if __name__ == '__main__':
    print("--------------一阶马尔科夫链---------------")
    # 定义状态转移矩阵 晴天 阴天 雨天
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.2, 0.3, 0.5]])

    # 定义初始状态分布，假设初始状态为晴天
    initial_state = np.array([1, 0, 0])  # 初始状态为 [1, 0, 0] 表示晴天
    print("初始状态分布矩阵：\n", initial_state)

    # 计算经过n天后各个状态的概率分布
    def predict_weather(n):
        state = initial_state
        for _ in range(n):
            state = np.dot(state, P)
        return state

    # 打印经过1天后的天气状态概率分布
    print("经过1天后的天气状态概率分布：", predict_weather(1))
    # 打印经过2天后的天气状态概率分布
    print("经过2天后的天气状态概率分布：", predict_weather(2))
    # 打印经过3天后的天气状态概率分布
    print("经过3天后的天气状态概率分布：", predict_weather(3))
    # 打印经过4天后的天气状态概率分布
    print("经过4天后的天气状态概率分布：", predict_weather(4))
