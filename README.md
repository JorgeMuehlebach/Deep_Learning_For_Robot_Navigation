# Deep_Learning_For_Robot_Navigation
Test PPO, DQN and GAIL using a simulated environment with a husky robot, obstacles and a goal. There are 5 different task difficulty levels that a model can be trained on. The tasks are described bellow: 
1. Task 1 involves a static husksy starting position and a fixed goal position.
2. Task 2 involves a random husky starting orientation and a random goal position.
3. Task 3 involves a random husky orientation and 1 obstacle between the husky and the goal (that is randomly positioned).
4. Task 4 involves a random husky orientation and a wall in between the husky and the goal.
5. Task 5 involves a random husky orientation and n randomly positioned obstacles. 

# installation
- ensure you have conda installed, see: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

clone the repo: 
```
git clone https://github.com/JorgeMuehlebach/Deep_Learning_For_Robot_Navigation.git
```

Create a conda environment using the environment.yml file 
```
conda env create -f environment.yml
```

activate the conda environment:
```
conda activate RobotNavigationML
```
# START TRAINING!

```
python TrainHuskyRL.py [task number e.g., 2]
```
As the model is training information about its progress will be printed to the console once it has trained sufficiently (e.g., success rate: 0.95) then you can stop the training and save the model by entering the trial letter. The trial letter by default is set to A, so pressing A will save the model to the results folder. There will also be data related to training saved to the stats file that can be used at your discretion. I personally used excel to make some fancy graphs to show how the success rate and reward increased throughout training. 

# Run the model
Show the husky completing the task you just trained it to complete by running the following command
```
python Running_A_Saved_Model.py [task number e.g., 2]
```
ensure that the task number is the same for both training and running 

# Changing the algorithm to DQN and PPO
To change the algorithm just open both the TrainHuskyRL.py and the Running_A_Saved_Model.py and change the parameters that are being passed into the main methods
