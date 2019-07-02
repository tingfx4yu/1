import mujoco_py
import gym
env = gym.make('Hopper-v2')  # or 'Humanoid-v2' 
while True:
    env.render()
