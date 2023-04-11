import numpy as np
from UAV_env import UAVEnv


class StateNormalization(object):
    def __init__(self):
        env = UAVEnv()
        # 规定状态的上限和下限：无人机电池, 位置, 剩余计算任务
        M = env.M
        self.high_state = np.array([5e5, env.ground_length, env.ground_width])
        self.high_state = np.append(self.high_state, np.ones(M * 2) * env.ground_length)
        self.high_state = np.append(self.high_state, np.ones(M) * 3145729)

        self.low_state = np.zeros(3 * M + 3)

    def state_normal(self, state):
        # 归一化处理
        return state / (self.high_state - self.low_state)
