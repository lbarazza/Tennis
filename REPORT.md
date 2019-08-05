# Report

## Algorithm
This project uses the DDPG algorithm starting off from the previous implementation in the [Reacher](https://github.com/lbarazza/Reacher) project and making some slight modifications. In fact, the only difference is the instatiation of two different agents instead of just one. To check out how the DDPG algorithm works in detail check the [report](https://github.com/lbarazza/Reacher/blob/master/REPORT.md) of the Reacher project.

To show how general the DDPG algorithm is, the same network structure and hyperparamters as the ones in the Reacher project have been used.

## Networks
Two separate networks for actor and critic networks have been used. Both networks have two hidden layers with 100 neurons each and ReLU activation. The actor network has a tanh activation function applied to the output. Instead, the critic network doesn't have any. These are the same networks as the ones used in the Reacher project, the only difference is that each agent has its own version of each one of these.

## Hyperparameters
These are the same networks as the ones used in the Reacher project.

|     Hyperparamter                          |      Value                      |
|--------------------------------------------|:-------------------------------:|
|    actor learning rate                     |          0.0005                 |
|    critic learning rate                    |          0.0005                 |
|    noise standard deviation start          |          0.15                   |
|    noise standard deviation end            |          0.025                  |
|    noise standard deviation decay frames * |          200000                 |
|    gamma                                   |          0.99                   |
|    networks update rate                    |          1                      |
|    tau                                     |          0.001                  |
|    memory length                           |          1,000,000              |
|    replay memory start                     |          1,000                  |
|    n. no op at beginning of training       |          1,000                  |
|    batch_size                              |          60                     |
|    replay_start_size                       |          1,000                  |

* the standard deviation has been decreased linearly in the amount of frames specified

## Results
The agent was able to solve the environment in 2249 episodes:
![alt text](https://raw.githubusercontent.com/lbarazza/Tennis/master/images/sample_test03_stats.png "DDPG stats")
(the average reward is in dark blue, while the actual single rewards are plotted in light blue)

## Improvements
The agent could be improved to:
- implement MADDPG which makes the environment more stationary by training the critic by showing it all the actions taken by all the other agents
- use Ornstein Uhlenbeck noise instead of normal gaussian noise with zero mean
- make use of Prioritized Experience Replay
