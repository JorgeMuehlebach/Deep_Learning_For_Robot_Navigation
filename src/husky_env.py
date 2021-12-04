import pybullet as p
import pybullet_data
import os, inspect
import time
import numpy as np
import gym
from gym import spaces
import math
import random
import sys
from HuskyInput import HuskyInputController 
from gym.utils import seeding
import math 
import sys
from pybullet_utils import bullet_client as bc
from BuildingComplexEnvironments import ComplexEnvironment

# the joint indexes for the Husky URDF file used to set the wheel speeds
front_left_wheel_index = 2
back_left_wheel = 4
front_right_wheel_index = 3
back_right_wheel = 5


class Husky(gym.Env):

    def __init__(self, isDiscrete=False, debug=True, action_repeat=24, renders=False, goal_radius=0.5, stage=5, num_obstacles = 30) -> None:
        """
        Args:
            isDiscrete (bool, optional): is the action space descrete or not?. Defaults to False.
            debug (bool, optional): setting to true shows lidar rays. Defaults to True.
            action_repeat (int, optional): The default timestep is 1/240 second (240hz) this can be adjusted with 
            setTimeStep or setPhysicsEngineParameter API, however it is not reccomended, so instead each step/action is 
            repeated n number of times. Defaults to 24 (effectively changing the timestep to 10hz).
            renders (bool, optional): weather or not you want the gui of the env to show, usually only done 
            to monitor if the husky and environment is as expected qualitatively. Defaults to False.
            goal_radius (float, optional): the size of the goal, making it larger makes the task easier. Defaults to 0.5.
            stage (int, optional): sets the stage number which is related to difficulty level (5 is the highest). Defaults to 5.
            num_obstacles (int, optional): the number of obstacles in the environment, only utilised if stage == 5. Defaults to 30.
        """
        self.stage = stage
        self.random_start_angle_robot = True
        self.wall = False
        self.obstacle_path ="house_small.urdf"
        self.terminate_upon_reaching_the_goal = False
        self.env_radius = 20
        self.num_rays = 0
        self.renders = renders
        self.timestep = 0.0001
        self.number_of_randomly_placed_obstacles = num_obstacles
        self.static_goal_position = False
        # used for logging logging of this data occurs in Callback.py
        self.SUCCESS_COUNT = 0
        self.FAILURE_COUNT = 0
        self.REWARD_PER_STEP = []
        self.REWARD_FOR_THIS_EPISODE = []
        self.REWARD_PER_EPISODE = []
        self.time_taken_for_generating = 0
        self.time_taken_for_lidar = 0
        self.time_taken_for_reset = 0
        self.time_taken_for_step = 0
        self.collided_time = 0
        self.sim_step_time = 0
        self.ray_obs = []

        if renders:
            self._p = bc.BulletClient(connection_mode=p.GUI)
        else: 
            self._p = bc.BulletClient()
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.isDiscrete = isDiscrete
        self.action_repeat = action_repeat
        self.action_space = []
        if isDiscrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(np.array([0,0]), np.array([1,1]), dtype=np.float32)

        self.envStepCounter = 0
        self.episode_counter = 0
        self.debug = debug
        self.ray_points = []
        self.ray_debugs = []

        self.goal_radius = goal_radius
        self.done = False
        self.ingoal = False
        self.orange_goal = -1
        self.fail = False
        # used for smoothness reward calculation
        self.previous_action = [0, 0]
        self.current_action = [0, 0]
        self.smoothness_reward = 0

        if stage == 1:
            self.random_start_angle_robot = False
            self.terminate_upon_reaching_the_goal = True
            self.static_goal_position = True
            self.number_of_randomly_placed_obstacles = 0
        elif stage == 2:
            self.random_start_angle_robot = True
            self.terminate_upon_reaching_the_goal = False
            self.number_of_randomly_placed_obstacles = 0
        elif stage == 3:
            self.random_start_angle_robot = True
            self.terminate_upon_reaching_the_goal = True
            self.num_rays = 20
            self.number_of_randomly_placed_obstacles = 0
        elif stage == 4:
            self.random_start_angle_robot = True
            self.terminate_upon_reaching_the_goal = False
            self.num_rays = 20
            self.number_of_randomly_placed_obstacles = 0
        else:
            self.random_start_angle_robot = True
            self.terminate_upon_reaching_the_goal = False
            self.num_rays = 20
            self.number_of_randomly_placed_obstacles = num_obstacles
        self.obstacle_builder = ComplexEnvironment(self._p, env_radius=self.env_radius, n_obstacles=self.number_of_randomly_placed_obstacles)
        low = [-math.pi, -self.env_radius*2]
        high = [math.pi, self.env_radius*2]
        low.extend(np.zeros(self.num_rays))
        high.extend(np.ones(self.num_rays))
        self.observation_space = spaces.Box(np.array(low), np.array(high), dtype=np.float32)
        self.env_radius = 20

    def step(self, action):
        start_step = time.time()
        real_action = [0, 0]
        if (self.isDiscrete):
            wheel_speeds = [[0,1], [1,1], [1, 0]]
            real_action = wheel_speeds[action]
        else:
            real_action = action
        self.current_action = real_action
        # apply action
        self.set_wheels_left_speed(real_action[0])
        self.set_wheels_right_speed(real_action[1])
        for i in range(self.action_repeat):
            # The default timestep is 1/240 second (240hz) this can be adjusted with 
            # setTimeStep or setPhysicsEngineParameter API, however it is not reccomended, so instead each step/action is 
            start_sim_step = time.time()
            self._p.stepSimulation()
            end_sim_step = time.time()
            self.sim_step_time += end_sim_step - start_sim_step
            if self.renders:
                time.sleep(self.timestep)

            self.envStepCounter += 1
        obs = self.get_obs()
        reward = self.get_reward()
        self.REWARD_PER_STEP.append(reward)
        self.REWARD_FOR_THIS_EPISODE.append(reward)
        self.done = self.is_done()
        self.fail = self.failed()
        self.ingoal = True if self.in_goal() else False
        if self.ingoal and not self.terminate_upon_reaching_the_goal:
            print("SUCCESS")
            self.move_goal()
        # just for logging
        elif self.ingoal:
            print("SUCCESS")
        if self.done:
            self.REWARD_PER_EPISODE.append(sum(self.REWARD_FOR_THIS_EPISODE))
            self.REWARD_FOR_THIS_EPISODE = []
        end_step = time.time()
        self.time_taken_for_step += end_step - start_step
        return obs, reward, self.done, {}

    def reset(self):
        start_reset = time.time()
        self.obstacles = []
        self._p.resetSimulation()
        plane = self._p.loadURDF("plane.urdf")
        self._p.setGravity(0, 0, -10)
        self.husky = self.create_husky(0,0)
        self.envStepCounter = 0
        self.orange_goal = -1
        self.set_goal_position()
        obs = self.get_obs()
        end_reset = time.time()
        self.time_taken_for_reset += end_reset - start_reset
        self.obstacle_builder.build(self._goal_position, self.obstacles)
        return obs

    def set_goal_position(self):
        self._goal_position = [0, 0, 0]
        # just sets the goal position for this env
        if self.static_goal_position:
            self._goal_position = [3, 4, 0]
            self.orange_goal = self._p.loadURDF("orange_goal.urdf", self._goal_position)
        if self.stage == 3:
            random_angle = 2 * math.pi * random.random()
            goal_distance = 8
            goal_x = goal_distance * math.cos(random_angle) 
            goal_y = goal_distance * math.sin(random_angle) 
            obstacle_distance = 4
            obstacle_x = (obstacle_distance) * math.cos(random_angle)
            obstacle_y = (obstacle_distance) * math.sin(random_angle)
            self._goal_position = [goal_x, goal_y, 0.5]
            self.orange_goal = self._p.loadURDF("orange_goal.urdf", self._goal_position)
            obstacle = self._p.loadURDF(self.obstacle_path, [obstacle_x,obstacle_y, 0.5])
            self.obstacles.append(obstacle)
        if self.stage == 4:
            self.build_wall_objects()
        else:
            self.move_goal()



    def move_goal(self):
        # finds a valid position for the goal that is not next to or on top of an obstacle
        if self.stage == 4:
            if self._goal_position[0] == -10:
                self._goal_position[0] = 0
            else:
                self._goal_position[0] = -10
            if self.orange_goal != -1:
                self._p.removeBody(self.orange_goal)
            self.orange_goal = self._p.loadURDF("orange_goal.urdf", self._goal_position)
            return 
        goal_placed = False
        while(not goal_placed):
            random_angle = 2 * math.pi * random.random()
            goal_distance = 2 + random.random()*10
            goal_x = goal_distance * math.cos(random_angle) 
            goal_y = goal_distance * math.sin(random_angle) 
            too_close = False
            for obs in self.obstacles:
                pos, orn = self._p.getBasePositionAndOrientation(obs)
                distance_to_new_obstacle = math.sqrt((pos[0]-goal_x)**2 + (pos[1]-goal_y)**2)
                if distance_to_new_obstacle < 2.5:
                    too_close = True
                    break
            if not too_close:
                self._goal_position = [goal_x, goal_y, 0]
                if self.orange_goal != -1:
                    self._p.removeBody(self.orange_goal)
                self.orange_goal = self._p.loadURDF("orange_goal.urdf", self._goal_position)
                goal_placed = True
            

    def build_wall_objects(self):
        self._goal_position =[-10, 0, 0]
        self.orange_goal = self._p.loadURDF("orange_goal.urdf", self._goal_position)
        side = -1
        if random.random() > 0.5:
            side = -1
        else:
            side= 1
        self._obstacle_position = [-5,2*side,0.5]
        wall_segment_y = 2*side
        for i in range(30):
            wall_segment_y -= side*1.1
            obstacle =self._p.loadURDF(self.obstacle_path, [self._obstacle_position[0],wall_segment_y,0.5])
            self.obstacles.append(obstacle)

    def render(self):
        return np.array([])

    def close(self):
       self._p.disconnect()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]       
        
    def test(self):
        self.get_obs()
        self.set_ray_points()
        if self.debug:
            self._display_ray()
            time.sleep(0.01)
            self._remove_debug_rays()
        else:
            time.sleep(0.01)
        if self.in_goal():
            self._p.removeBody(self.orange_goal)
            self.green_goal =self._p.loadURDF("green_goal.urdf", self._goal_position)

        
    def get_reward(self):
        reward = 0
        huskyPos, orn =self._p.getBasePositionAndOrientation(self.husky)
        distance_to_goal = math.sqrt((self._goal_position[0] - huskyPos[0])**2 + (self._goal_position[1] - huskyPos[1])**2)
        distance_from_origin = math.sqrt(huskyPos[0]**2 + huskyPos[1]**2)
        closest_obstacle = 9999
        for ray in self.ray_obs:
            if ray < closest_obstacle:
                closest_obstacle = ray
        too_close_reward = (1/0.9)*closest_obstacle - 0.111
        angle_cost = -abs(self.get_angle_to_goal())/math.pi + 1
        distance_cost = 1/(distance_to_goal +1)

        action_difference = math.sqrt((self.current_action[0] - self.previous_action[0])**2 + (self.current_action[1] - self.previous_action[1])**2)

        action_smoothness_reward = (-1/(math.sqrt(2)))*action_difference + 1
        self.smoothness_reward = action_smoothness_reward

        coef_distance, coef_angle, coef_obstacle_proximity, coef_smoothness = 2, 1, 0, 0
        reward = coef_distance*distance_cost + coef_angle*angle_cost + coef_obstacle_proximity*too_close_reward + coef_smoothness*action_smoothness_reward
        reward_normalised = reward/(coef_distance + coef_angle + coef_obstacle_proximity + coef_smoothness)
        if self.in_goal():
            reward_normalised += 100
        if self.collided():
            reward_normalised = -100
        if distance_from_origin > 20:
            reward_normalised = -100
        self.previous_action = self.current_action
        return reward_normalised



    def collided(self):

        start = time.time()
        closest_obstacle = 999999
        for ray in self.ray_obs:
            if ray < closest_obstacle:
                closest_obstacle = ray
        if closest_obstacle < 0.09:
            end = time.time()
            return True
        else:
            end = time.time()
            self.collided_time += end - start
            return False
        
    
    def get_obs(self):
        obs = []
        huskyPos, orn =self._p.getBasePositionAndOrientation(self.husky)

        distance_to_goal = math.sqrt((self._goal_position[0] - huskyPos[0])**2 + (self._goal_position[1] - huskyPos[1])**2)
        start = time.time()
        self.set_ray_points()
        end = time.time()
        self.time_taken_for_lidar += end - start
        angle_to_goal = self.get_angle_to_goal() 
        obs.extend([angle_to_goal, distance_to_goal])
        self.ray_obs = []

        #TestStableBaselines
        if self.debug:
            self._display_ray()
            time.sleep(0.0001)
            self._remove_debug_rays()

        from_points, to_points = self.ray_points
        for from_position, to_position in zip(from_points, to_points):
            results =self._p.rayTest(from_position, to_position)
            # addding the fraction of the way the ray has travelled 
            # 1 being 100% of the rays distance, 0 being an obstacle right infront of it
            self.ray_obs.append(results[0][2])
        obs.extend(self.ray_obs)
        obs = np.array(obs)

        return obs
        
    def is_done(self):
        huskyPos, orn =self._p.getBasePositionAndOrientation(self.husky)
        distance_from_origin = math.sqrt(huskyPos[0]**2 + huskyPos[1]**2)
        if self.collided() or self.envStepCounter > 20000 or distance_from_origin > 20 or (self.terminate_upon_reaching_the_goal and self.in_goal()):
            if self.failed():
                # printed if the huksy hits an obstacle or goes out of bounds
                print("FAILED")
            self.episode_counter +=1
            return True
        else:
            return False

    def failed(self):
        huskyPos, orn =self._p.getBasePositionAndOrientation(self.husky)
        distance_from_origin = math.sqrt(huskyPos[0]**2 + huskyPos[1]**2)
        if self.collided() or distance_from_origin > 20:
            return True
        else:
            return False


    def get_random_goal_position(self):
        x = random.randint(-10,10)
        y = random.randint(-10,10)
        return [x, y, 0]

    def create_tree(self, x, y):
        tree =self._p.loadURDF("../tree.urdf",[x,y,0.5])

    def create_house(self, x,y):
        house = self._p.loadURDF("house.urdf", [x,y,1])
        return house

    def create_husky(self,x,y):
        if self.random_start_angle_robot:
            random_husky_angle = random.uniform(-math.pi, math.pi)
            quaternion_angle =self._p.getQuaternionFromEuler([0, 0, random_husky_angle])
            return self._p.loadURDF("husky/husky.urdf",[x, y, 0], quaternion_angle)
        else:
            return self._p.loadURDF("husky/husky.urdf", [x, y, 0])

        

    def in_goal(self):
        # returns true if the husky is inside the goal circle
        # and false otherwise 
        huskyPos, orn =self._p.getBasePositionAndOrientation(self.husky)
        distance = math.sqrt((self._goal_position[0] - huskyPos[0])**2 + (self._goal_position[1] - huskyPos[1])**2)
        if distance < self.goal_radius:
            return True
        else:
            return False 

    def set_wheels_left_speed(self, speed):
        # use a range of 0-1 for speed 1 being 6.399 kph or 10 radians per second
        maxVel = 10  #rad/s
        maxForce = 100  #Newton
        targetVel = maxVel*speed
        self._p.setJointMotorControl2(self.husky, front_left_wheel_index,self._p.VELOCITY_CONTROL, targetVelocity=targetVel, force=maxForce)
        self._p.setJointMotorControl2(self.husky, back_left_wheel,self._p.VELOCITY_CONTROL, targetVelocity=targetVel, force=maxForce)

    def set_wheels_right_speed(self, speed):
        # use a range of 0-1 for speed 1 being 6.399 kph or 10 radians per second
        maxVel = 10  #rad/s
        maxForce = 100  #Newton
        targetVel = maxVel*speed
        self._p.setJointMotorControl2(self.husky, front_right_wheel_index,self._p.VELOCITY_CONTROL, targetVelocity=targetVel, force=maxForce)
        self._p.setJointMotorControl2(self.husky, back_right_wheel,self._p.VELOCITY_CONTROL, targetVelocity=targetVel, force=maxForce)

    def set_ray_points(self):
        basePos, orn =self._p.getBasePositionAndOrientation(self.husky)
        start = 0
        end = 10
        # increase_from_180 is used if you would like more than 180 degree lidar
        increase_from_180 = 0
        scan_range = np.pi + increase_from_180
        # starting angle
        angle = -self._p.getEulerFromQuaternion(orn)[2] - increase_from_180/2
        increment = scan_range/(self.num_rays -1) if self.num_rays > 1 else 0

        from_points = []
        to_points = []
        for i in range(self.num_rays):
            from_points.append([
                basePos[0] + start * np.sin(angle),
                basePos[1] + start * np.cos(angle),
                0.6
            ]) 

            to_points.append([
                basePos[0] + end * np.sin(angle),
                basePos[1] + end * np.cos(angle),
                0.6
            ])
            angle += increment
        self.ray_points = np.array(from_points), np.array(to_points)

    def _display_ray(self):
        from_points, to_points = self.ray_points
        i = 0
        for from_position, to_position in zip(from_points, to_points):
            results =self._p.rayTest(from_position, to_position)
            color = [0, 1, 0]
            if results[0][2] < 0.9:
                color = [1,0,0]
            debug_ray = self._p.addUserDebugLine(from_position, to_position, lineColorRGB = color, lifeTime=0.01)
            self.ray_debugs.append(debug_ray)
            i+=1

    def _remove_debug_rays(self):
        for ray in self.ray_debugs:
            self._p.removeUserDebugItem(ray)
        self.ray_debugs = []
        import numpy as np

    def get_angle_to_goal(self):
        ''' compute angle (in degrees) for p0p1p2 corner
        Inputs:
            p0,p1,p2 - points in the form of [x,y]
        '''
        basePos, orn =self._p.getBasePositionAndOrientation(self.husky)
        angle1 = -self._p.getEulerFromQuaternion(orn)[2] + np.pi/2
        p0 = [basePos[0] + 10*np.sin(angle1), basePos[1] + 10*np.cos(angle1)]
        p1 = [basePos[0], basePos[1]]
        p2 = [self._goal_position[0], self._goal_position[1]]
        if p2 is None:
            p2 = p1 + np.array([1, 0])
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        return angle

if __name__ == "__main__":
    '''
    calling this file from the commpand line will enable you to view the husky in the env
    and control it with the arrow keys
    takes the task number as an argument default is task 5
    '''
    if len(sys.argv) > 1 and sys.argv[1].isnumeric():
        env = Husky(debug=True, renders=True, stage=int(sys.argv[1]))
    else:
        env = Husky(debug=True, renders=True)
    
    env.reset()
    # allows you to control the husky with the arrowkeys
    husky_controller = HuskyInputController(env)
    step_counter = 0
    ep_reward = 0
    while(1):
        if env.is_done():
            break
        env.test()
        if env.in_goal():
            env.move_goal()
        for i in range(24):
            env._p.stepSimulation()
    env._p.disconnect()



