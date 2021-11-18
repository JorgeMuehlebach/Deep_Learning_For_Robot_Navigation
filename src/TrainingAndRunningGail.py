import gym
import time
from stable_baselines import GAIL, SAC, PPO2
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from husky_env import Husky
from stable_baselines.common import make_vec_env
from Callback import CustomCallback
import matplotlib.pyplot as plt
import pandas as pd
trial_letter = 'C'
# create vectorised husky env for ppo2 load
vec_env = make_vec_env(Husky, n_envs=1, env_kwargs={'debug':False, 'renders':False, 'isDiscrete':True})
# load a trained model
trained_rl_model = PPO2.load("Husky_result" + trial_letter, env=vec_env) #env=env
# generate expert trajectories based off trained model
generate_expert_traj(trained_rl_model, 'expert_husky' + trial_letter, n_episodes=400)
# load the trajectories into the dataset variables
dataset = ExpertDataset(expert_path='expert_husky'+ trial_letter +'.npz' , traj_limitation=400, verbose=1)
# initialise husky env
env = Husky(debug=False, renders=False, isDiscrete=True, goal_radius=1)
# initialise gail model
model = GAIL('MlpPolicy', env, dataset, verbose=1)
# trains it
trial_letter = 'C'
callback = callback=CustomCallback(stopping_letter=trial_letter, using_ppo=False)
model.learn(total_timesteps=6000000, callback=callback)
# save the model
model.save("gail_husky_recent")
df = pd.DataFrame(callback.stats_each_episode, columns=['n', 'time', 'reward', 'moving action smoothness','smoothness', 'moving_avg', 'success_moving_avg'])
df.to_csv('stats_' + trial_letter + ".csv")
model.save("Husky_result" + trial_letter)

debug_env = Husky(debug=True, renders=True, isDiscrete=True, goal_radius=1)
obs = debug_env.reset()
while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = debug_env.step(action)
  time.sleep(0.01)