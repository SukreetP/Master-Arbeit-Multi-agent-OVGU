# Multi-Agent Reinforcement Learning for Deadlock Handling in Automated Guided Vehicle Systems

This repository contains the implementation developed for my Master’s thesis at Otto von Guericke University Magdeburg. The thesis investigates how Multi-Agent Reinforcement Learning can be used to handle deadlock situations in an Automated Guided Vehicle system operating inside a logistical warehouse simulation.

The project combines a Siemens Tecnomatix Plant Simulation model with Python-based Deep Q-Network agents. Three independent AGV agents interact with the simulation, observe their local states, choose movement actions, and learn policies that reduce deadlocks, collisions, and waiting behavior over repeated training episodes.

## Thesis Objective

The main objective of this work is to develop and evaluate a generic Multi-Agent Reinforcement Learning approach for resolving deadlocks in warehouse logistics systems. The approach uses independent Deep Q-Learning agents that interact with a dynamic Plant Simulation environment and learn suitable actions under operational constraints.

The work focuses on:

* Modeling a multi-AGV warehouse environment in Siemens Plant Simulation
* Connecting the simulation environment to Python through Plant Simulation RemoteControl
* Implementing independent DQN agents for multiple AGVs
* Training agents under deadlock-capable scenarios
* Comparing standard DQN and prioritized replay buffer variants
* Evaluating performance using reward, collision count, waiting time, drive time, idle time, and loading/unloading time

## System Overview

```text
+-----------------------------+
| Siemens Plant Simulation    |
| Warehouse / AGV Model (.spp)|
+-------------+---------------+
              |
              | RemoteControl COM Interface
              v
+-----------------------------+
| Python Gym-like Environment |
| env.py                      |
+-------------+---------------+
              |
              | State, Reward, Done
              v
+-----------------------------+
| Multi-Agent DQN Training    |
| Agent 1 | Agent 2 | Agent 3 |
+-------------+---------------+
              |
              | Actions: forward / backward / stop
              v
+-----------------------------+
| Simulation Update + Metrics |
| rewards, collisions, times  |
+-----------------------------+
```

## Core Idea

Each AGV is treated as an independent reinforcement learning agent. At each decision step, every AGV receives a state representation from the Plant Simulation model and selects one of three actions:

```text
0 -> Move forward
1 -> Move backward
2 -> Stop
```

The simulation then executes the selected actions, updates the AGV positions, calculates rewards, and returns the next state. The agents use Deep Q-Networks to approximate action-value functions and improve their decisions over time.

## Main Components

### Plant Simulation Model

The `.spp` files contain the warehouse and AGV simulation models created in Siemens Tecnomatix Plant Simulation. The simulation contains AGV movement logic, reward variables, collision counters, waiting-time measurements, and object update methods.

### Custom Environment

`env.py` provides a Gym-style wrapper around the Plant Simulation model. It exposes:

* `step()` for applying actions to all AGVs
* `reset()` for retrieving the current AGV state
* `Table()` for reading the state table from Plant Simulation
* `render()` for making the Plant Simulation model visible
* `close()` for closing the simulation session

### DQN Agents

`DQN_agent_1.py`, `DQN_agent_2.py`, and `DQN_agent_3.py` define the neural networks used by the three AGV agents.

The basic network architecture is:

```text
Input state
 -> Dense(64) + sigmoid activation
 -> Dense(32) + softmax activation
 -> Dense(number_of_actions)
 -> Q-values for each action
```

The models use the Huber loss function and Adam optimizer.

## Experiments

| Experiment folder                                            | Purpose                                                                     |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `DQN training agents single part carry experiment_v_local`   | Baseline multi-agent DQN experiment for the single-part-carry scenario      |
| `DQN training agents seperate reward (with load time)`       | DQN experiment with separate reward design including loading/unloading time |
| `DQN prioritized replay buffer single part carry experiment` | DQN experiment using prioritized experience replay                          |
| `DQN prioritized replay buffer alpha,beta 0.5`               | Prioritized replay experiment with alpha and beta set to 0.5                |

## Training Parameters

| Parameter                |             Value |
| ------------------------ | ----------------: |
| Number of AGV agents     |                 3 |
| Actions per agent        |                 3 |
| Replay buffer size       |            30,000 |
| Training start threshold | 1,000 transitions |
| Batch size               |                32 |
| Discount factor gamma    |             0.999 |
| Initial epsilon          |               1.0 |
| Minimum epsilon          |            0.0005 |
| Epsilon decay            |             10e-6 |
| Learning rate            |              1e-4 |
| Training epochs          |               100 |
| Steps per epoch          |             2,000 |

## Requirements

This project is designed for a Windows environment because Siemens Plant Simulation is controlled through the Windows COM interface.

### Software

* Windows 10 or newer
* Siemens Tecnomatix Plant Simulation with RemoteControl support
* Python 3.9 or 3.10 recommended
* A valid Plant Simulation license

### Python Packages

```bash
pip install numpy matplotlib gym tensorflow keras comtypes pywin32
```

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install numpy matplotlib gym tensorflow keras comtypes pywin32
```

## Configuration

Before running the training script, update the Plant Simulation model path inside `modelversion_new.py`.

Example:

```python
com_obj = win32.Dispatch("Tecnomatix.PlantSimulation.RemoteControl.22.1")
com_obj.loadModel(r"C:\path\to\your\single_part_test.spp")
```

The current scripts contain local absolute paths from the original thesis development machine. These paths must be changed before running the project on another computer.

## Running an Experiment

Open PowerShell or Command Prompt, activate the environment, and move into one experiment folder:

```powershell
cd "DQN training agents single part carry experiment_v_local"
python modelversion_new.py
```

The script will:

1. Start the Plant Simulation RemoteControl interface
2. Load the selected `.spp` model
3. Initialize the three DQN agents
4. Train the agents using the simulation environment
5. Append training metrics to `.txt` log files
6. Save trained model weights as `.h5` files

Important: The scripts append results to the existing `.txt` files. For a clean experiment, rename or delete old log files before starting a new run.

## Plotting Results

To generate or update plots, run the plotting scripts inside an experiment folder or its `plots/` folder.

Examples:

```powershell
python reward_plot.py
python collision_plot.py
```

For experiments with additional plot scripts:

```powershell
cd plots
python drive_time_plot.py
python load_unload_time.py
python network_loss.py
```

## Outputs and Evaluation Metrics

Each training run stores numerical logs in `.txt` files. These files are used to analyze the behavior of the AGV agents over time.

| File                            | Description                               |
| ------------------------------- | ----------------------------------------- |
| `Rewards_train_a1.txt`          | Reward history for AGV 1                  |
| `Rewards_train_a2.txt`          | Reward history for AGV 2                  |
| `Rewards_train_a3.txt`          | Reward history for AGV 3                  |
| `collision_train.txt`           | Number of collisions per training epoch   |
| `wait_a*_time.txt`              | Waiting time for each AGV                 |
| `drive_time_a*.txt`             | Driving time for each AGV                 |
| `loading_unloading_time_a*.txt` | Loading and unloading time for each AGV   |
| `idle_time_a*.txt`              | Idle time for each AGV                    |
| `loss_history_a*.txt`           | Neural-network training loss for each AGV |
| `dqn_AGVdeadlock_*.h5`          | Saved DQN model weights                   |

Some included logs show clear improvement during training. For example, in the baseline single-part-carry experiment, the final logged rewards reach approximately 7350 for AGV 1, 7750 for AGV 2, and 7400 for AGV 3, while the logged collision count decreases to around 12 near the end of the run.

## Methodology Summary

The thesis follows the Independent Q-Learning approach for multi-agent reinforcement learning. Each AGV learns its own Q-function while sharing the same simulation environment with the other agents. This creates a dynamic learning problem where one agent’s action can affect the future state and reward of the others.

The workflow is:

1. Initialize the Plant Simulation model
2. Read the current AGV state table
3. Select actions using epsilon-greedy exploration
4. Execute AGV actions in Plant Simulation
5. Read rewards, next states, and performance metrics
6. Store transitions in replay memory
7. Train DQN models from sampled transitions
8. Update target networks
9. Repeat over many simulation steps and epochs
10. Compare the trained policies across experiment variants

## Limitations

* The project depends on Siemens Plant Simulation and Windows COM automation.
* The simulation model path is hard-coded in the training scripts and must be updated manually.
* The implementation is designed around three AGVs and three discrete actions.
* The experiment folders contain repeated versions of similar files; future work could refactor the code into shared modules.
* Training logs are appended to existing text files, so old logs should be cleared before new experiments.
* The repository contains generated files such as `__pycache__` and trained model weights; these can be cleaned or managed with `.gitignore` for a public release.

## Future Improvements

Possible improvements include:

* Refactoring common DQN, environment, and plotting code into reusable modules
* Adding a `requirements.txt` or `environment.yml` file
* Replacing hard-coded file paths with configuration files or command-line arguments
* Adding automated experiment tracking with TensorBoard, MLflow, or CSV summaries
* Supporting additional AGV counts and warehouse layouts
* Comparing DQN with Double DQN, Dueling DQN, PPO, A2C, or centralized MARL methods
* Improving reward shaping to better balance collisions, waiting time, travel time, and throughput
* Adding unit tests for the replay buffer and environment interface
* Creating a lightweight mock environment for testing without Plant Simulation

## Academic Context

This repository was developed as part of the Master’s thesis:

**Development of a Multi-Agent Reinforcement Learning Approach to Deal with Deadlocks in an Automated Guided Vehicle System**

Author: **Sukreet Pal**
University: **Otto von Guericke University Magdeburg**
Program: **M.Sc. Digital Engineering**

## License

This repository is shared for academic and reference purposes. If you use, modify, or refer to the code, methodology, results, figures, or thesis material from this project, please cite the original thesis and acknowledge the author.
Suggested citation:
Sukreet Pal, Development of a Multi-Agent Reinforcement Learning Approach to Deal with Deadlocks in an Automated Guided Vehicle System, Master’s Thesis, Otto von Guericke University Magdeburg, 2023.

## Contact

For questions about the project, thesis implementation, or experimental setup, please contact the repository author.
