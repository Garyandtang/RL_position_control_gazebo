## 2020-10-14Configuration

| Env Model         | Differential wheeled model                                   |
| ----------------- | ------------------------------------------------------------ |
| Maximum linear V  | 1                                                            |
| Maximum angular V | 3                                                            |
| reward in running | ($c_r\times (d_{t-1}-d_t)-k_1\times (v_{t}-v_{t-1}) -k_2\times (w_t - w_{t-1})$) |
| cr, K1, K2        | [200, 0,0]                                                   |
| Terminal reward   | ($r_t$)                                                      |
| rt, k3,k4         | [50, 0,0]                                                    |

| Agent Model | PPO                                                          |
| ----------- | ------------------------------------------------------------ |
| Actor       | nn.Linear(state_dim, 64),<br/>                nn.ReLU(),<br/>                nn.Linear(64, 32),<br/>                nn.ReLU(),<br/>                nn.Linear(32, 32),<br/>                nn.ReLU(),<br/>                nn.Linear(32,2),<br/>                nn.Tanh() |
| Critic      | self.state_block = nn.Sequential(<br/>            nn.Linear(state_dim, 64),<br/>            nn.ReLU()<br/>        )<br/>        self.outpu_block = nn.Sequential(<br/>            nn.Linear(action_dim+64, 64),<br/>            nn.ReLU(),<br/>            nn.Linear(64, 32),<br/>            nn.ReLU(),<br/>            nn.Linear(32, 1),<br/>            nn.Linear(1,1)<br/>        )<br/>    def forward(self, state, action):<br/>        state_feature = self.state_block(state)<br/>        merged = torch.cat((state_feature, action), dim=1)<br/>        output = self.outpu_block(merged)<br/>        return output |

### Result

| Training Episodes | Result                                   |
| ----------------- | ---------------------------------------- |
| 100               | cannot, flip                             |
| 200               | cannot, unstable                         |
| 300               | cannot                                   |
| 400               | cannot                                   |
| 500               | arrive at 1st target                     |
| 600               | finish 2 targets                         |
| 700               | out of boundary                          |
| 800               | out of boundary                          |
| 900               | out of boundary, almost flip             |
| 1000              |                                          |
| 1500              | out of boundary                          |
| 1800              | finished all avg 504 (rota, jump, dance) |
| 1900              | finished all avg 324                     |
| 2000              | cannot, (rotate at start point)          |



## 2020-10-11

### Configuration

| Env Model         | Differential wheeled model             |
| ----------------- | -------------------------------------- |
| Maximum linear V  | (1) 1.1                                |
| Maximum angular V | (3) 3.14                               |
| reward in running | ($c_r\times (d_{t-1}-d_t)$)            |
| cr, K1, K2        | [200, 5,8]                             |
| Terminal reward   | ($r_t-k_3\times v_{t} -k_4\times w_t$) |
| rt, k3,k4         | [50, 5, 8]                             |

| Agent Model | PPO                                                          |
| ----------- | ------------------------------------------------------------ |
| Actor       | nn.Linear(state_dim, 64),<br/>                nn.ReLU(),<br/>                nn.Linear(64, 32),<br/>                nn.ReLU(),<br/>                nn.Linear(32, 32),<br/>                nn.ReLU(),<br/>                nn.Linear(32,2),<br/>                nn.Tanh() |
| Critic      | self.state_block = nn.Sequential(<br/>            nn.Linear(state_dim, 64),<br/>            nn.ReLU()<br/>        )<br/>        self.outpu_block = nn.Sequential(<br/>            nn.Linear(action_dim+64, 64),<br/>            nn.ReLU(),<br/>            nn.Linear(64, 32),<br/>            nn.ReLU(),<br/>            nn.Linear(32, 1),<br/>            nn.Linear(1,1)<br/>        )<br/>    def forward(self, state, action):<br/>        state_feature = self.state_block(state)<br/>        merged = torch.cat((state_feature, action), dim=1)<br/>        output = self.outpu_block(merged)<br/>        return output |

### Result

| Training Episodes | Result             |
| ----------------- | ------------------ |
| 100               | avg: 114           |
| 200               | 215.6              |
| 300               | 147                |
| 400               | 235                |
| 500               | 126.6              |
| 600               | 327                |
| 700               | 253.92307692307693 |
| 800               | 317                |
| 900               |                    |
| 1000              | 129                |
| 1100              | flip               |
| 1300              | cannot             |
| 1400              | 166                |
| 2100              | 446.6              |
| 2500              | out of boundary    |
| 3000              | 695                |
| 4000              | 504                |
| 7400              | cannot             |

### Figure

#### 100 episodes

![Figure_1](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/100_episodes.png)

#### 200 episodes

![200_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/200_episodes.png)

#### 300 episodes

![300_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/300_episodes.png)

#### 400 episodes

![400_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/400_episodes.png)

#### 500 episodes

![500_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/500_episodes.png)

#### 600 episodes

![600_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/600_episodes.png)

#### 700 episodes

![700_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/700_episodes.png)

#### 800 episodes

![800_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/800_episodes.png)

#### 1000 episodes

![1000_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/1000_episodes.png)

#### 1400 episodes

![1400_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/1400_episodes.png)

#### 2100 episodes

![2100_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/2100_episodes.png)

#### 3000 episodes

![3000_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/3000_episodes.png)

#### 4000 episodes

![4000_episodes](/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_11_diff_wheel_64_32/img/4000_episodes.png)

