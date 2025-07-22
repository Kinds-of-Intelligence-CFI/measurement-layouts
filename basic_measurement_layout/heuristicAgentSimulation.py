import csv
import fnmatch
import math
import numpy as np
import os
import re
import random

from PIL import Image

from fastcore.script import call_parse
from animalai.environment import AnimalAIEnvironment
from animalai.actions import AAIActions, AAIAction

class cameraBraitenberg():
    """Implements a simple Braitenberg vehicle agent that heads towards food"""
    def __init__(self, random_action_rate):
        self.actions = AAIActions()
        self.GOODGOAL = np.array([0.7372549, 0.8784314, 0.5411765], dtype=np.float32)
        self.random_action_rate = random_action_rate

    def get_action_camera(self, obs) -> AAIAction:
        """Returns the action to take given the current visual observation"""
        newAction = self.actions.LEFT
        if self.ahead(obs, self.GOODGOAL):
            newAction = random.choices([self.actions.FORWARDS, self.actions.NOOP], weights=[1 - self.random_action_rate, self.random_action_rate], k=1)[0]
        elif self.left(obs, self.GOODGOAL):
            newAction = random.choices([self.actions.LEFT, self.actions.NOOP], weights=[1 - self.random_action_rate, self.random_action_rate], k=1)[0]
        elif self.right(obs, self.GOODGOAL):
            newAction = random.choices([self.actions.RIGHT, self.actions.NOOP], weights=[1 - self.random_action_rate, self.random_action_rate], k=1)[0]
        else:
            newAction = self.actions.LEFT
        return newAction


    def ahead(self, obs, object):
        """Returns true if the input object is ahead of the agent"""
        middle = obs[:, obs.shape[1]//2-1:obs.shape[1]//2+1, :]
        if np.any(np.all(middle == object, axis=-1)):
            return True
        else:
            return False

    def left(self, obs, object):
        """Returns true if the input object is left of the agent"""
        left_half = obs[:, :obs.shape[1]//2, :]
        if np.any(np.all(left_half == object, axis=-1)):
            return True
        else:
            return False


    def right(self, obs, object):
        """Returns true if the input object is right of the agent"""
        right_half = obs[:, obs.shape[1]//2:, :]
        if np.any(np.all(right_half == object, axis=-1)):
            return True
        else:
            return False
        
def find_yaml_files(directory):
    yaml_files = []
    task_names = []
    
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.yml') + fnmatch.filter(filenames, '*.yaml'):
            yaml_files.append(os.path.join(root, filename))
            task_names.append(filename)
    
    return yaml_files, task_names


def regex_value_finder(string):
    pattern = re.compile(r'x_(\d+)_z_(\d+)_size_([\d]+[.][\d])')

    match = pattern.search(string)

    if match:
        x_value = int(match.group(1))
        z_value = int(match.group(2))
        size_value = float(match.group(3))

        return x_value, z_value, size_value
    else:
        print("No match found.")

def runBraitenbergAndStore(agent: cameraBraitenberg, 
                           pixel_input: int,
                           config_folder: str, 
                           results_path: str, 
                           env_path: str,
                           agent_inference: bool = True):
    
    random.seed(2025)

    yaml_files, task_names = find_yaml_files(config_folder)


    for yaml, name in zip(yaml_files, task_names):
        aai_env = None
        while aai_env is None:
            try:
                port = random.randint(1000, 20000)
                aai_env = AnimalAIEnvironment( 
                        inference=agent_inference, #Set true when watching the agent
                        seed = 2023,
                        worker_id=port,
                        file_name=env_path,
                        arenas_configurations=yaml,
                        base_port=port,
                        useCamera=True,
                        useRayCasts = False,
                        resolution = pixel_input,
                        no_graphics=False,
                        timescale = 1 if agent_inference else 300,
                    )
            except:
                pass
        
        behavior = list(aai_env.behavior_specs.keys())[0] # by default should be AnimalAI?team=0

        firststep = True

        if firststep:
            aai_env.step() # take first step to get an observation
            firststep = False
                    
        dec, term = aai_env.get_steps(behavior)

        done = False

        episodeReward = 0

        step_counter = 0
    
        while not done:
            observations = aai_env.get_obs_dict(dec.obs)["camera"]
            
            observations = np.transpose(observations, (1, 2, 0))

            action = agent.get_action_camera(observations)

            aai_env.set_actions(behavior, action.action_tuple)

            aai_env.step()

            step_counter += 1

            dec, term = aai_env.get_steps(behavior)

            if len(dec.reward) > 0 and len(term) <= 0:
                episodeReward += dec.reward
            elif len(term) > 0: #Episode is over
                episodeReward += term.reward

                done = True
                firststep = True
            else:
                pass
        
        file_exists = os.path.isfile(results_path)
        with open(results_path, 'a' if file_exists else 'w', newline='') as csv_file:
            csv_write = csv.writer(csv_file)
            if not file_exists:
                csv_write.writerow(['agent', 'pixelInput', 'navigationNoise', 'episode', 'x_value', 'z_value', 'distance', 'size', 'finalReward'])
                csv_file.flush()
                
            x, z, size = regex_value_finder(str(name))

            distance = math.sqrt((abs(x-20))**2 + (abs(z-0.5))**2)
            agent_name = f"Agent_{str(pixel_input)}"
            csv_write.writerow([str(agent_name), pixel_input, str(agent.random_action_rate), str(name), x, z, distance, size, episodeReward[0]])
            csv_file.flush()
        aai_env.close()


@call_parse
def main(pixel_input: int, 
         randomness: float,
         config_folder: str, 
         results_path: str, 
         env_path: str,
        ):
    agent_all = cameraBraitenberg(random_action_rate = randomness)
    
    runBraitenbergAndStore(agent=agent_all,
                           pixel_input=pixel_input, 
                           config_folder=config_folder, 
                           results_path=results_path, 
                           env_path=env_path,
                           agent_inference=False)
    
    print(f"Finished running Braitenberg agent with pixel input {pixel_input} and randomness {randomness}. Results stored in {results_path}.")
