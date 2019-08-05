from pathlib import Path
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
from ddpg_agent import DDPGAgent

env = UnityEnvironment(file_name="Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

agent1 = DDPGAgent(nS=state_size,
                  nA=action_size,
                  lr_actor=0.0005,
                  lr_critic=0.0005,
                  gamma=0.99,
                  batch_size=60,
                  tau=0.001,
                  memory_length=int(1e6),
                  no_op=int(1e3),
                  net_update_rate=1,
                  std_initial=0.15,#0.15
                  std_final=0.025,
                  std_decay_frames=200000)

agent2 = DDPGAgent(nS=state_size,
                  nA=action_size,
                  lr_actor=0.0005,
                  lr_critic=0.0005,
                  gamma=0.99,
                  batch_size=60,
                  tau=0.001,
                  memory_length=int(1e6),
                  no_op=int(1e3),
                  net_update_rate=1,
                  std_initial=0.15,
                  std_final=0.025,
                  std_decay_frames=200000)

# setup csv and checkpoint files
run_name = 'sample_test04' # name of the current test

checkpoint_file_path1 = 'checkpoints/' + run_name + '_1' + '.tar'
checkpoint_file_path2 = 'checkpoints/' + run_name + '_2' + '.tar'
csv_file_path = 'csv/' + run_name + '.csv'
update_rate = 1

checkpoint_file1 = Path(checkpoint_file_path1)
checkpoint_file2 = Path(checkpoint_file_path2)
if checkpoint_file1.is_file() and checkpoint_file2.is_file():
    start_episode = agent1.load(checkpoint_file_path1) + 1
    agent2.load(checkpoint_file_path2)
else:
    start_episode = 0

csv_file = Path(csv_file_path)

# add headers to csv file
if not csv_file.is_file():
    data_frame=pd.DataFrame(data={
                                    'episode': [],
                                    'reward': [],
                                    'average reward': []
                                 },
                            columns = ['episode', 'reward', "average reward"])
    data_frame.to_csv(csv_file_path)

rets = deque(maxlen=100)
for episode in range(10000):
    
    ret = 0
    data = {
            'episode': [],
            'reward': [],
            'average reward': []
           }

    # show a preview from time to time while training
    env_info = env.reset(train_mode=(not episode%100==0))[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    while True:
        action1 = agent1.choose_action(states[0])
        action2 = agent2.choose_action(states[1])
        actions = np.vstack((action1, action2))
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        agent1.step(states[0], actions[0], rewards[0], next_states[0], dones[0])
        agent2.step(states[1], actions[1], rewards[1], next_states[1], dones[1])
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break

    score = np.max(scores)
    # print out results
    rets.append(score)
    average_reward = np.mean(rets)
    print('Episode: {}\t Score: {}\t Avg. Reward: {}\t N. Steps: {}\t Std.: {}'.format(episode, score, average_reward, agent1.nSteps, agent1.std))

    if average_reward >= 0.5:
        print("\t--> SOLVED! <--\t")
        break

    data['episode'].append(episode)
    data['reward'].append(score)
    data['average reward'].append(average_reward)

    # save stats and checkpoint
    if episode%update_rate==0:
        agent1.save(checkpoint_file_path1, episode)
        agent2.save(checkpoint_file_path2, episode)
        data_frame = pd.DataFrame(data=data)
        data_frame.set_index('episode')
        with open(csv_file_path, 'a') as f:
            data_frame.to_csv(f, header=False)


env.close()
