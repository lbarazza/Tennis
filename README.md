# Tennis
DDPG for solving a multi-agent environment


## Dependencies
This project is implemented using Python 3.6, PyTorch, NumPy and the UnityEnvironment package. For the plotting part of the project Matplotlib is used, while for the part dealing with csv, Pandas is used.

## Installation
Download the repository with

```
git clone https://github.com/lbarazza/Reacher
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

To visualize an already existing agent loaded from a checkpoint, change the path to the desired checkpoint in the file and run (to visualize a pre-trained agent use checkpoint path "checkpoints/sample_test2.tar"):

```
python visualize_agents.py
```
