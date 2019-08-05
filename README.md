# Tennis
![alt text](https://raw.githubusercontent.com/lbarazza/Tennis/master/images/tennis_ex_image.png "Tennis image")

## Project Details
This project solves the environment provided in the Udacity Reinforcement Learning Nanodegree Program. The system consists of two AI agents that need to learn to pass the tennis ball to each other without dropping it. If an agent hits the ball over the net, it receives a reward of +0.1, if it drops it, it receives a reward of -0.01, thus the two agents are learning to collaborate and not compete. Each agent receives an observation of consisting of 24 differen variables. The possible actions for each agent are to go towards/away from the net and jumping (both continous). By having two agents, two different scores are available, the score that is considered to represent the episode is the highest one of the two. The environemnt is considered solved when the average reward over the last 100 episodes reaches 0.5.

## Dependencies
This project is implemented using Python 3.6, PyTorch, NumPy and the UnityEnvironment package. For the plotting part of the project Matplotlib is used, while for the part dealing with csv, Pandas is used.

## Installation
Download the repository with

```
git clone https://github.com/lbarazza/Tennis
```

To install the environment, download the version corresponding to your operating system:

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)

[Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)

[Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

[Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Unzip the downloaded file and place it in the root folder of the project.
Now move into the project folder and create a virtual environement where we can install all of the dependencies

```
cd Tennis
conda create --name tennis-project python=3.6 matplotlib pandas
```

Activate the virtual environment:

(Mac and Linux)
```
source activate tennis-project
```

(Windows)
```
activate tennis-project
```

Now install all the dependencies:

```
pip install -r requirements.txt
```

## Run the Code
To create a new agent and make it learn from scratch run (or resume training):

```
python main.py
```

To visualize an already existing agent loaded from a checkpoint, change the path to the desired checkpoint in the file and run (to visualize a pre-trained agent use checkpoint path "checkpoints/sample_test03.tar"):

```
python visualize_agents.py
```
