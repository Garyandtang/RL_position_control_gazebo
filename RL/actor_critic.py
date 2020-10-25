'''
reference on concatenate layer: https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462
reference on IROS 2017 work: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8202134
paper title: Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation
'''
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from diff_wheel_env_new import GazeboEnv
import numpy as np
import logging
import time
file_name = 'testing'
log = logging.getLogger("myapp") 
logging.basicConfig(filename='test.log', level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
# pytorch implement of IROS 2017 works
class ActorBlock(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorBlock, self).__init__()
        self.linear_v = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32,1),
                nn.Tanh()
                )
        self.angular_v = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32,1),
                nn.Tanh()
                )
    def forward(self, state):
        x1 = self.linear_v(state)
        x2 = self.angular_v(state)
        x = torch.cat((x1, x2), dim=1)
        return x

class CriticBlock(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(CriticBlock, self).__init__()
        self.state_block = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.outpu_block = nn.Sequential(
            nn.Linear(action_dim+64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Linear(1,1)
        )
    def forward(self, state, action):
        state_feature = self.state_block(state)
        merged = torch.cat((state_feature, action), dim=1)
        output = self.outpu_block(merged)
        return output
        
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  ActorBlock(state_dim, action_dim)
        # critic
        self.critic = CriticBlock(state_dim, action_dim)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state,action)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            # if is_terminal:
            #     discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            print("discounted_reward is {}".format(discounted_reward))
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            info = "Loss is {:.4f}".format(loss.mean())
            print(info)
            # log.warning("woot")
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        

            
if __name__ == '__main__':
    # main()
    logging.info("start training")
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 6000        # max timesteps in one episode
    
    update_timestep = 40      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 40               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
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
    # ppo.policy_old.load_state_dict(torch.load('./trained_model/PPO_position_control_64.pth'))
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # smoothing the action
    action_prev = [0.0,0.0]
    p = 0.8 # changing factor

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        print("{}th episodes training".format(i_episode))
        # print("state type: {}".format(type(state[2])))
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            
            
            action_dict = dict(linear_vel=p*action[0]+(1-p)*action_prev[0], angular_vel=p*action[1]+(1-p)*action_prev[1])
            # action_dict = dict(linear_vel=action[0], angular_vel=action[1])
            # print(action_dict)
            action_prev = [action_dict['linear_vel'], action_dict['angular_vel']]
            state, reward, done, _ = env.excute(action_dict)
            print("State:{}, reward: {}, r_v: {}, l_v: {}".format(state, reward, action[0], action[1]))
            # print("reward: {}".format(reward))
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            # if render:
            #     env.render()
            
            if done:
                logging.info("Successfully arrive at destination")
                print("Successfully arrive at destination")
                break
        
        avg_length += t
        
        # # stop training if avg_reward > solved_reward
        # if running_reward > (log_interval*solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        #     break
        
        # save every 500 episodes
        if i_episode % 100 == 0:
            # torch.save(ppo.policy.state_dict(), './PPO_position_control_diff_wheel_64_32_{}_{}_{}_{}.pth'.format(time.localtime(time.time()).tm_mon, time.localtime(time.time()).tm_mday,time.localtime(time.time()).tm_mday, i_episode))
            torch.save(ppo.policy.state_dict(), './PPO_position_control_diff_wheel_64_32_{}_{}_{}_{}.pth'.format(time.localtime(time.time()).tm_mon, time.localtime(time.time()).tm_mday,time.localtime(time.time()).tm_mday, i_episode))
            logging.info('save model at {}'.format(i_episode))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {}  Avg length: {}  Avg reward: {}'.format(i_episode, avg_length, running_reward))
            logging.info('Episode {}  Avg length: {}  Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    print("Finished {} episode training".format(max_episodes))
    logging.info('Finished')