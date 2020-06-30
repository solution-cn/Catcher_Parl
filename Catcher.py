#-*- coding = utf-8 -*-
#@Time : 2020/6/29 17:38
#@Author : solution
#@File : watergame.py
#@Software: PyCharm
from ple.games import Catcher
from ple import PLE

import os
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger


LEARNING_RATE = 1e-3


class Model(parl.Model):
    def __init__(self, act_dim):
        hid_size=128
        self.fc1 = layers.fc(size=hid_size, act='tanh')
        self.fc1 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs): 

        out = self.fc1(obs)

        return out


from parl.algorithms import PolicyGradient 


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program): 
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program): 
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0) 
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost



def run_episode(p, agent):
    eval_step = 0
    obs_list, action_list, reward_list = [], [], []
    p.reset_game()
    while True:
        eval_step += 1
        info = p.getScreenRGB()
        obs = dealing(info)
        obs_list.append(obs)
        action = agent.sample(obs) 
        action_list.append(action)
        action = p.getActionSet()[action]
        reward= p.act(action)
        reward_list.append(reward)
        if p.game_over():
            break
    return obs_list, action_list, reward_list


def evaluate(p, agent):
    eval_reward = []
    for i in range(5):
        p.reset_game()
        eval_step = 0
        episode_reward = 0
        while True:
            eval_step += 1
            info = p.getScreenRGB()
            obs = dealing(info)
            action = agent.sample(obs)
#             action = agent.predict(obs)
            action = p.getActionSet()[action]
            reward= p.act(action)
            episode_reward += reward
            if p.game_over():
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def dealing(image):
    """ 预处理 128x128x3 uint8 frame into 4096 (64x64) 1维 float vector """
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image != 0] = 1 
    return image.astype(np.float).ravel()


def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


game = Catcher(width=128, height=128)
p = PLE(game, fps=30, display_screen=True)
p.init()
obs_dim = 64*64
act_dim = len(p.getActionSet())
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

# 加载模型
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')
    logger.info("------Loading------")

for i in range(1000):
    obs_list, action_list, reward_list = run_episode(p, agent)
    if i % 10 == 0:
         logger.info("Train Episode {}, Reward Sum {}.".format(i,sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)
    agent.learn(batch_obs, batch_action, batch_reward)

    if (i + 1) % 100 == 0:
        total_reward = evaluate(p, agent)
        logger.info('Episode {}, Test reward: {}'.format(i + 1,total_reward))

#save the parameters to ./model.ckpt
agent.save('./model.ckpt')
# total_reward = evaluate(p, agent)
# logger.info('Episode {}, Test reward: {}'.format(1000,total_reward))
