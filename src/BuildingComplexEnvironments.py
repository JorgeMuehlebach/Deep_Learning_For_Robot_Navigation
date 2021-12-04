import math
import random
import pybullet as p
from pybullet_utils import bullet_client as bc
class ComplexEnvironment:
    def __init__(self, pybullet, robot_location=[0,0], env_radius=20, n_obstacles=10, seperation=2.5) -> None:
        self.seperation = seperation
        self.robot_location = robot_location
        #self.goal_location = goal_location
        self.env_radius = env_radius
        self.n_obstacles = n_obstacles
        self.p = pybullet
        self.obstacles = []
        self.obstacle_postions = []
        
    
    def build(self, goal_position, obstacles, build=True):
        '''
            adds all the obstacles to the environment into positions at which they will not collide
        '''
        self.obstacle_postions = []
        self.obstacles = obstacles
        for i in range(self.n_obstacles):
            obstacle_placed_in_env = False
            placement_attempts = 0
            while(not obstacle_placed_in_env):
                random_angle = 2 * math.pi * random.random()
                random_distance_from_centre = self.env_radius * random.random()
                obstacle_x = random_distance_from_centre * math.cos(random_angle)
                obstacle_y = random_distance_from_centre * math.sin(random_angle)
                away_from_other_obstacles = True
                distance_to_robot_location = math.sqrt((self.robot_location[0] - obstacle_x)**2 + (self.robot_location[1] - obstacle_y)**2)
                distance_to_goal_location = math.sqrt((goal_position[0] - obstacle_x)**2 + (goal_position[1] - obstacle_y)**2)
                for obs in self.obstacles:
                    pos, orn = self.p.getBasePositionAndOrientation(obs)
                    distance_to_new_obstacle = math.sqrt((pos[0]-obstacle_x)**2 + (pos[1]-obstacle_y)**2)
                    if distance_to_new_obstacle < self.seperation:
                        away_from_other_obstacles = False
                        break
                if away_from_other_obstacles and distance_to_robot_location > self.seperation and distance_to_goal_location > self.seperation:
                    # place the obstacle in the environment 
                    random_obstacle_orientation = random.uniform(-math.pi, math.pi)
                    quaternion_angle = self.p.getQuaternionFromEuler([0, 0, random_obstacle_orientation])
                    if build:
                        obstacle = self.p.loadURDF("house_small.urdf", [obstacle_x,obstacle_y,0.5], quaternion_angle)
                        self.obstacles.append(obstacle)
                    self.obstacle_postions.append([obstacle_x, obstacle_y])
                    obstacle_placed_in_env = True
                    break
                placement_attempts +=1
                if placement_attempts > 2000:
                    return False    
                    #raise Exception("could not place obstacles, no valid locations")
        return True

    



            
            
            










    