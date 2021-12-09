

from husky_env import Husky
import pandas as pd
from stable_baselines.common.env_checker import check_env
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as MlpPolicyDQN
from stable_baselines import DQN
from stable_baselines import PPO2
import datetime
from Callback import CustomCallback
import math
from stable_baselines.common import make_vec_env
import matplotlib.pyplot as plt
import sys

def main(dqn=False, ppo2=False, continue_training=False, stage=5):
  """
  Args:
      dqn (bool, optional): using DQN or not. Defaults to False.
      ppo2 (bool, optional): using PPO or not. Defaults to False.
      continue_training (bool, optional): start training using a model 
      that has already been trained somewhat, ensure that the model was using the same env. Defaults to False.
      stage (integer, optional): (difficulty level) see the readme
  """
  trial_letter = 'F'
  if ppo2:
    env = make_vec_env(Husky, n_envs=1, env_kwargs={'debug':False, 'renders':False, 'isDiscrete':False, 'stage':stage})
  else:
    env = Husky(debug=False, renders=False, isDiscrete=False, stage=stage)
  
  if continue_training:
    if ppo2:
      model = PPO2.load("results/Husky_result" + trial_letter, env=env)
    else:
      model = DQN.load("results/Husky_result" + trial_letter, env=env)
  else:
    if ppo2:
      model = PPO2(MlpPolicy, env, verbose=2)
    else:
      model = DQN(MlpPolicyDQN, env, verbose=2)
  
  callback = callback=CustomCallback(stopping_letter=trial_letter, using_ppo=ppo2)
  model.learn(total_timesteps=10000000, callback=callback)
  df = pd.DataFrame(callback.stats_each_episode, columns=['n', 'time', 'reward', 'moving action smoothness','smoothness', 'moving_avg', 'success_moving_avg'])
  df.to_csv('results/stats_' + trial_letter + ".csv")
  model.save("results/Husky_result" + trial_letter)

  
if __name__ == '__main__':
    # this is the stage (difficulty level) see the readme
    stage = 5
    # can also be set with a command line argument
    if len(sys.argv) > 1 and sys.argv[1].isnumeric():
        stage=int(sys.argv[1])
    # by typing this character into the terminal while the the training is occuring
    # it will stop the training and save the model with the name Husky_result_[trial_letter]
    os.makedirs("results", exist_ok=True)
    main(dqn=False, ppo2=True, continue_training=False, stage=stage)
