import numpy as np
import win32com.client as win32
from env import CustomEnv
from DQN_agent_1 import DQN1
from DQN_agent_2 import DQN2
from DQN_agent_3 import DQN3
from typing import Deque
import random
import collections
import time
import matplotlib.pyplot as plt
from pathlib import Path

mod_path = Path(__file__).parent

com_obj = win32.Dispatch("Tecnomatix.PlantSimulation.RemoteControl.22.1")
com_obj.loadModel("Master-Arbeit-Multi-agent-OVGU\\test_marl.spp".format(mod_path))
com_obj.setVisible(True)
com_obj.SetTrustModels(True)

env = CustomEnv(com_obj)

class Agent:
    def __init__(self,env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.observations_a1 = self.env.state_agent1.shape
        self.observations_a2 = self.env.state_agent2.shape
        self.observations_a3 = self.env.state_agent3.shape
        self.states_a1 = []
        self.states_a2 = []
        self.states_a3 = []
        self.states_next_a1 =[]
        self.states_next_a2 = []
        self.states_next_a3 = []
        self.actions_a1 = self.env.action_space_a1.n
        self.actions_a2 = self.env.action_space_a2.n
        self.actions_a3 = self.env.action_space_a3.n
        # DQN Agent Variables
        self.replay_buffer_size = 300000
        self.train_start = 500
        self.memory1 = collections.deque(maxlen=self.replay_buffer_size)
        self.memory2 = collections.deque(maxlen=self.replay_buffer_size)
        self.memory3 = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.999
        self.epsilon = 1.0
        #self.epsilon2 = 0.5
        #self.epsilon3 = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DQN Network Variables
        self.state_shape_a1 = self.observations_a1
        self.state_shape_a2 = self.observations_a2
        self.state_shape_a3 = self.observations_a3
        self.learning_rate = 1e-4
        self.model1 = DQN1(self.state_shape_a1, self.actions_a1, self.learning_rate)
        self.model2 = DQN2(self.state_shape_a2, self.actions_a2, self.learning_rate)
        self.model3 = DQN3(self.state_shape_a3, self.actions_a3, self.learning_rate)
        self.target_model_1 = DQN1(self.state_shape_a1, self.actions_a1, self.learning_rate)
        self.target_model_2 = DQN2(self.state_shape_a2, self.actions_a2, self.learning_rate)
        self.target_model_3 = DQN3(self.state_shape_a3, self.actions_a3, self.learning_rate)
        self.target_model_1.update_model(self.model1)
        self.target_model_2.update_model(self.model2)
        self.target_model_3.update_model(self.model3)
        self.batch_size = 32

    def get_action_agent_1(self, state_a1):
        rand = np.random.rand()
        #print("Action:", np.argmax(self.model1.predict(state_a1)))
        #print(rand,self.epsilon)
        #laura1 = np.argmax(self.model1.predict(state_a1))
        if rand <= self.epsilon:

            #return np.argmax(np.random.randint(self.actions_a1))
            return env.action_space_a1.sample()
        else:
            #print("Action:", np.argmax(self.model1.predict(state_a1)))
            return np.argmax(self.model1(state_a1))



    def get_action_agent_2(self, state_a2):
        rand = np.random.rand()
        #print(rand,self.epsilon)
        #print("Action:", np.argmax(self.model2.predict(state_a2)))
        if rand <= self.epsilon:
            #return np.argmax(np.random.randint(self.actions_a2))
            return env.action_space_a2.sample()
        else:
            #print("Action:", np.argmax(self.model(state_a2)))
            return np.argmax(self.model2(state_a2))

    def get_action_agent_3(self, state_a3):
        rand = np.random.rand()
        #print("Action:", np.argmax(self.model1.predict(state_a3)))
        #print(rand,self.epsilon)
        if rand <= self.epsilon:
            #return self.actions_a3
            #return np.argmax(np.random.randint(self.actions_a3))
            return env.action_space_a3.sample()
        else:
            #print("Action:", np.argmax(self.model3(state_a3)))
            return np.argmax(self.model3(state_a3))

    def train(self, epochs, num_of_steps):

        mem1 = []
        mem2 = []
        best_reward_mean = -50000000.0
        last_rewards: Deque = collections.deque(maxlen=10)

        for j in range(epochs):

            com_obj.resetSimulation(".Models.Model")
            com_obj.startSimulation(".Models.Model")

            eps = []
            rews = []
            n_coll = []
            time.sleep(0.25)

            for episode in range(1, num_of_steps + 1):

                #if self.epsilon > self.epsilon_min and len(self.memory1) and len(self.memory2) and len(self.memory3):
                    #self.epsilon *= self.epsilon_decay

                while True:
                    state = env.reset()

                    state_a1 = state[0].reshape(1,6)
                    state_a2 = state[1].reshape(1,6)
                    state_a3 = state[2].reshape(1,6)

                    action_a1 = self.get_action_agent_1(state_a1)
                    action_a2 = self.get_action_agent_2(state_a2)
                    action_a3 = self.get_action_agent_3(state_a3)
                    # print(action_a1)
                    a1_next_state, a2_next_state, a3_next_state, reward, done, info = env.step(action_a1, action_a2,
                                                                                               action_a3)
                    self.remember(state_a1, state_a2, state_a3, a1_next_state, a2_next_state, a3_next_state, action_a1,
                                  action_a2, action_a3, reward, done)
                    self.replay()
                    com_obj.executeSimTalk(".Models.Model.AGVPOOL.updateVeh11(.UserObjects.AGV1)")
                    com_obj.executeSimTalk(".Models.Model.AGVPOOL.updateVeh21(.UserObjects.AGV2)")
                    com_obj.executeSimTalk(".Models.Model.AGVPOOL.updateVeh31(.UserObjects.AGV3)")
                    n_collisions = com_obj.getValue('.Models.Model.n_collisions')
                    Reward_AGV1 = com_obj.getValue(".Models.Model.RewardAGV1")
                    Reward_AGV2 = com_obj.getValue(".Models.Model.RewardAGV2")
                    Reward_AGV3 = com_obj.getValue(".Models.Model.RewardAGV3")

                    total_reward = com_obj.getValue(".Models.Model.RL")

                    eps.append(episode)
                    rews.append(total_reward)
                    n_coll.append(n_collisions)

                    self.target_model_1.update_model(self.model1)
                    self.target_model_2.update_model(self.model2)
                    self.target_model_3.update_model(self.model3)

                    print(f"Step: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    break

            last_rewards.append(total_reward)

            current_rewards_mean = np.mean(last_rewards)

            if current_rewards_mean > best_reward_mean:
                best_reward_mean = current_rewards_mean


                self.model1.save_model("{}\\dqn_AGVdeadlock_1.h5".format(mod_path))
                self.model2.save_model("{}\\dqn_AGVdeadlock_2.h5".format(mod_path))
                self.model3.save_model("{}\\dqn_AGVdeadlock_3.h5".format(mod_path))
                print(f"New best mean: {best_reward_mean}")


            mem1.append(total_reward)
            file1 = open("Rewards.txt", "a")
            print(total_reward, file=file1)
            file1.close()

            mem2.append(n_collisions)
            file2 = open("collision.txt", "a")
            print(n_collisions, file=file2)
            file2.close()

    def remember(self, a1_state,a2_state,a3_state,next_state_a1,next_state_a2,next_state_a3,action_a1,action_a2,action_a3, reward, done):
        #print("remember:",a1_state)
        #print("action:",action_a1)
        #print("next_state:",next_state_a1)
        #print("reward:",reward)
        self.memory1.append((a1_state, action_a1,next_state_a1, reward, done))
        self.memory2.append((a2_state, action_a2, next_state_a2, reward, done))
        self.memory3.append((a3_state, action_a3, next_state_a3, reward, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def replay(self):
        if len(self.memory1) and len(self.memory2) and len(self.memory3) < self.train_start:
            return

        minibatch1 = random.sample(self.memory1, self.batch_size)
        #print("Minibatch :",minibatch1[0][0][0])
        #print(minibatch1)
        minibatch2 = random.sample(self.memory2, self.batch_size)
        #print(minibatch2)
        minibatch3 = random.sample(self.memory3, self.batch_size)
        #print(minibatch3)
        a1_state,action_a1,states_next_a1,rewards,dones = zip(*minibatch1)
        a2_state, action_a2,states_next_a2,rewards, dones = zip(*minibatch2)
        a3_state, action_a3, states_next_a3,rewards, dones = zip(*minibatch3)



        #,a2_state,a3_state, action_a1,action_a2,action_a3,rewards, states_next_a1,states_next_a2,states_next_a3, dones = zip(*minibatch)
        #print(zip(*minibatch))
        # [s1, s2, s3, s4, s5]
        # mit concatenate: np.array [[s1],[s2], ...]
        states_a1 = np.vstack(a1_state).astype(np.int32)
        states_a2 = np.vstack(a1_state).astype(np.int32)
        states_a3 = np.vstack(a1_state).astype(np.int32)
        #states = np.concatenate(a1_state,a2_state,a3_state)
        #print(states_next_a1)
        states_next_a1 =np.vstack(states_next_a1).astype(np.int32)
        states_next_a2 =np.vstack(states_next_a2).astype(np.int32)
        states_next_a3 =np.vstack(states_next_a3).astype(np.int32)

        #print(states_a1,type(self.states_a1))

        q_values_a1 = self.model1(states_a1)
        q_values_a2 = self.model2(states_a2)
        q_values_a3 = self.model3(states_a3)
        #print(states_next_a1.shape)
        #print(states_next_a1)
        q_values_next_a1 = self.target_model_1(states_next_a1)
        q_values_next_a2 = self.target_model_2(states_next_a2)
        q_values_next_a3 = self.target_model_3(states_next_a3)
        #print("q_values:", q_values_a1, "q_values_next:", q_values_next_a2)
        for i in range(self.batch_size):
            a = action_a1[i]
            b = action_a2[i]
            c = action_a3[i]
            done = dones[i]
            if done:
                q_values_a1[i][a] = rewards[i]
                q_values_a2[i][b] = rewards[i]
                q_values_a3[i][c] = rewards[i]
            else:
                q_values_a1[i][a] = rewards[i] + self.gamma * np.max(q_values_next_a1[i])
                q_values_a2[i][b] = rewards[i] + self.gamma * np.max(q_values_next_a2[i])
                q_values_a3[i][c] = rewards[i] + self.gamma * np.max(q_values_next_a3[i])

        self.model1.fit(states_a1, q_values_a1)
        self.model2.fit(states_a2, q_values_a2)
        self.model3.fit(states_a3, q_values_a3)

if __name__ == "__main__":
    env = CustomEnv(com_obj)
    agent = Agent(env)
    agent.train(epochs=1000, num_of_steps=2000)
    env.close()
