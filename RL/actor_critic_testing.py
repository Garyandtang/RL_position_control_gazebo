'''
reference on concatenate layer: https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462
reference on IROS 2017 work: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8202134
paper title: Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation
'''
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from gazebo_env import GazeboEnv
import numpy as np
import logging
from actor_critic import ActorBlock, CriticBlock, ActorCritic, PPO, Memory
file_name = 'ppo_position_control'
logging.basicConfig(filename=file_name, level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
def main():
    ############## Hyperparameters ##############
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 150000        # max timesteps in one episode
    
    update_timestep = 40      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    # creating environment
    env = GazeboEnv()
    env.reset()
    state_dim = env.state.shape[0]
    action_dim = len(env.actions)
    
    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load('./trained_model/PPO_position_control2-10.pth'))
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    state = env.reset()
    # test once 
    for t in range(max_timesteps):
        action = ppo.select_action(state, memory)
        action_dict = dict(linear_vel=action[0], angular_vel=action[1])
        state, reward, done, _ = env.excute(action_dict)
        # ep_reward += reward
       
        if done:
            break
    print("Total time step: {}".format(t))
    # # testing loop
    # for i_episode in range(1, max_episodes+1):
    #     state = env.reset()
    #     for t in range(max_timesteps):
    #         time_step +=1
    #         # Running policy_old:
    #         action = ppo.select_action(state, memory)
            
            
    #         action_dict = dict(linear_vel=action[0], angular_vel=action[1])
    #         state, reward, done, _ = env.excute(action_dict)
    #         print("reward: {}".format(reward))
    #         # Saving reward and is_terminals:
    #         memory.rewards.append(reward)
    #         memory.is_terminals.append(done)
            
    #         # update if its time
    #         if time_step % update_timestep == 0:
    #             ppo.update(memory)
    #             memory.clear_memory()
    #             time_step = 0
    #         running_reward += reward
    #         # if render:
    #         #     env.render()
    #         if done:
    #             logging.info("Successfully arrive at destination")
    #             print("Successfully arrive at destination")
    #             break
        
    #     avg_length += t
        
    #     # # stop training if avg_reward > solved_reward
    #     # if running_reward > (log_interval*solved_reward):
    #     #     print("########## Solved! ##########")
    #     #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
    #     #     break
        
    #     # save every 500 episodes
    #     if i_episode % 50 == 0:
    #         torch.save(ppo.policy.state_dict(), './PPO_position_control.pth')
            
    #     # logging
    #     if i_episode % log_interval == 0:
    #         avg_length = int(avg_length/log_interval)
    #         running_reward = int((running_reward/log_interval))
            
    #         print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
    #         logging.info('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
    #         running_reward = 0
    #         avg_length = 0
    # print("Finished {} episode training".format(max_episodes))
    # logging.info("Finished {} episode training".format(max_episodes))
            
if __name__ == '__main__':
    main()