"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""

import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

# 超参数
#####################  hyper parameters  ####################
MAX_EPISODES = 500
# MAX_EPISODES = 50000

# LR_A = 0.000001  # learning rate for actor
# LR_C = 0.000002  # learning rate for critic
# LR_A = 0.001  # learning rate for actor
# LR_C = 0.002  # learning rate for critic
LR_A = 0.1  # learning rate for actor
LR_C = 0.2  # learning rate for critic
GAMMA = 0.001  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
OUTPUT_GRAPH = False


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# 训练完成后进行评价,训练的函数不用再调learn方法.没有数据集的概念
def eval_policy(ddpg, eval_episodes=10):
    # eval_env = gym.make(env_name)
    eval_env = UAVEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.
    # 10个回合?
    for i in range(eval_episodes):
        state = eval_env.reset()
        # while not done:
        for j in range(int(len(eval_env.UE_loc_list))):
            action = ddpg.choose_action(state)
            action = np.clip(action, *a_bound)
            state, reward = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        # 没有设置存储器满的标签

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # 4层网络
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    # 没有save和restore函数：用于保存训练好的参数以及使用训练好的参数进行评价。


###############################  training  ####################################
np.random.seed(2)
tf.set_random_seed(2)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 1  # control exploration
# var = 0.1  # control exploration
# var = 0.01  # control exploration
t1 = time.time()

# 回合奖励list、回合耗时list、回合耗能list
ep_reward_list = []
ep_time_list = []
ep_energy_list = []
s_normal = StateNormalization()

steps = []

for i in range(MAX_EPISODES):
    # 回合开始重置环境
    s = env.reset()
    # 回合奖励
    ep_reward = 0
    ep_energy = 0
    ep_time = 0

    # 循环参数step：一个step代表一个时间帧
    j = 0
    while j < MAX_EP_STEPS:
        start = time.time()
        # ddpg获取的action,ddpg在learn时更新参数,传参为环境返回的reward等信息。动作为连续动作。
        a = ddpg.choose_action(s_normal.state_normal(s))
        # clip函数的使用，机械臂中防止超出范围
        # todo：var变量的作用:类似于更新频率dt?动态变化。
        a = np.clip(np.random.normal(a, var), *a_bound)  # 动作添加高斯噪声，目的是为了进行探索
        # 关键部分：环境的反馈（6个值，多了3个,均为布尔值,分别代表3个异常分支）
        s_, r, is_terminal, offloading_ratio_change, reset_dist, energy = env.step(a, i)
        # 根据布尔指标调整a[]参数：action
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        # 依然只存储4个值。前一个状态，执行动作a，变为下一个状态，获取的回报为r
        ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍

        # 超出容量进行学习，这一部分可以定义一个标志位memory_full进行判断
        if ddpg.pointer > MEMORY_CAPACITY:
            var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            ddpg.learn()
        s = s_
        # 累加获取到的reward和delay
        ep_reward += r
        ep_energy += energy
        if j == MAX_EP_STEPS - 1 or is_terminal:
            # ep_reward是一回合结果
            ep_time = time.time() - start
            # if not is_terminal:
            #     ep_energy = 35
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, ' Explore: %.3f' % var,
                  ' Done:', is_terminal, ' ep_energy:', ep_energy)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            ep_time_list = np.append(ep_time_list, ep_time)
            ep_energy_list = np.append(ep_energy_list, ep_energy)
            # 输出文件
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        # 用于step循环
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
################# reward训练结果 ######################
# plt.plot(ep_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Reward")

################# 耗时训练结果 ######################
# plt.plot(ep_time_list)
# plt.xlabel("Episode")
# plt.ylabel("time")

################# 耗能训练结果 ######################
plt.plot(ep_energy_list)
plt.xlabel("Episode")
plt.ylabel("energy")

plt.show()
