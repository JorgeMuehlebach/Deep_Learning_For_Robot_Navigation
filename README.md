# Deep_Learning_For_Robot_Navigation
See https://stable-baselines.readthedocs.io/en/master/ for the algorithm implentations used for PPO, DQN and GAIL.


Allows testing of PPO, DQN and GAIL using a simulated environment with a husky robot, obstacles and a goal. There are 5 different navigation task difficulty levels that a model can be trained on. The tasks are described below: 
1. Task 1 involves a static husksy starting position and a fixed goal position.
2. Task 2 involves a random husky starting orientation and a random goal position.
3. Task 3 involves a random husky orientation and 1 obstacle between the husky and the goal (that is randomly positioned).
4. Task 4 involves a random husky orientation and a wall in between the husky and the goal.
5. Task 5 involves a random husky orientation and n randomly positioned obstacles. 

You can have a look at the tasks by running the husky_env.py file after you have completed the installation process with:
```
python husky_env.py [task number e.g., 5]
```
you can use the arrow keys to move around the environment as well. 

# installation

## Tested version
This repository tested with the following versions and it's not guaranteed to work with other versions.

| Software | Version |
| --- | --- |
| Ubuntu | 18.04.5 LTS |
| Linux Kernel | 5.4.0-65-generic |
| OpenAI gym | 0.18.3|
| Stable-baselines | 2.10.2 |
| OpenAI gym | 0.18.0 |
| Python | 3.7.10 |
| PyBullet | 3.1.7 |
| Tensorflow | 1.15.0 |
| Tensorboard | 1.15.0 |

Detail package version information can be found from [environment.yml](environment.yml)

- ensure you have conda installed, see: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

clone the repo: 
```
git clone https://github.com/JorgeMuehlebach/Deep_Learning_For_Robot_Navigation.git
```

Create a conda environment with python3.7

```
conda create --name RobotNavigationML python=3.7
```

and activate the conda environment:

```
conda activate RobotNavigationML
```
(Note that `RobotNavigationML` is prefixed in your terminal console (e.g., `(RobotNavigationML) username$path` which means the conda env is activated and all packages will be installed in that env)

manually install packages
```
pip install pybullet==3.1.7 tensorflow==1.15.0 gym==0.18.3 (more packages will be added)
```

Optionally one can also use [environment.yml](environment.yml) to create a conda environment (with your own risk) as follow:

Create a conda environment using the environment.yml file 
```
conda env create -f environment.yml
```

# START TRAINING!

```
cd src
```

```
python TrainHuskyRL.py [task number e.g., 2]
```
As the model is training, information about its progress will be printed to the console. Once it has trained sufficiently (e.g., success rate: 0.95) then you can stop the training and save the model by entering the trial letter. The trial letter by default is set to A (capital A), so pressing A will save the model to the results folder. There will also be data related to training saved to the stats file that can be used at your discretion. I personally used excel to make some fancy graphs to show how the success rate and reward increased throughout training. 

# Run the model
Show the husky completing the task you just trained it to complete by running the following command
```
python Running_A_Saved_Model.py [task number e.g., 2]
```
ensure that the task number is the same for both training and running 

# Changing the algorithm to DQN and PPO
By default it uses PPO to solve the tasks. To change the algorithm to DQN just open both the TrainHuskyRL.py and the Running_A_Saved_Model.py and change the parameters that are being passed into the main methods.
 To use GAIL run the following command:
 ```
 python TrainingAndRunningGail.py
 ```
 in order for this command to work you must have an already trained model available for the task you wish to train it on. Then you must go into the TrainingAndRunningGail.py file and modify the path to that file along with which task difficulty it is. GAIL takes about 5 hours to train on my computer and only works for task 2. I would be interested if others could get it working for more diffult tasks. 
