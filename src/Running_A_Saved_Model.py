"""Runs a model that you have trained, it will show it solving the task in a GUI
"""

from numpy.lib.function_base import angle

from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
#from Husky import Husky
#from envs.husky_wall import Husky
from husky_env import Husky
from stable_baselines import GAIL
from stable_baselines import DQN
import sys

def main(model_type="PPO2", trial_letter = "A", stage=5, num_steps=50000):
    """
    Args:
        model_type (str, optional): what model type you are loading. Defaults to "PPO2".
        trial_letter (str, optional): loads a model with the suffix trial letter, see TrainHuskyRl
        . Defaults to "A".
    """
    
    if model_type =="PPO2" or model_type =="GAIL":
        env = make_vec_env(Husky, n_envs=1, env_kwargs={'debug':False, 'renders':True, 'isDiscrete':False, 'goal_radius':1, 'stage':stage})
        if model_type =="GAIL":
            model = GAIL.load("results/gail_husky" + trial_letter, env=env)
        else:
            print("here")
            model = PPO2.load("results/Husky_result" + trial_letter, env=env)
    elif model_type=="DQN":
        env = Husky(debug=True, renders=True, isDiscrete=False, stage=stage)
        model = DQN.load("results/Husky_result" + trial_letter, env=env)
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done if model_type == "DQN" else done[0]:
            obs = env.reset()

if __name__ == '__main__':
    # this is the stage (difficulty level) see the readme
    stage = 5
    # can also be set with a command line argument
    if len(sys.argv) > 1 and sys.argv[1].isnumeric():
        stage=int(sys.argv[1])
    main(model_type="PPO2", trial_letter = "A", stage=stage, num_steps=50000)
