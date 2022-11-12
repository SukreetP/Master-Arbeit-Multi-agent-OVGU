import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from comtypes.client import CreateObject
from pathlib import Path
import numpy as np
import win32com.client as win32
import time


class CustomEnv(Env):
    def __init__(self, plant_sim):
        self.action_space_a1 = Discrete(3)
        self.action_space_a2 = Discrete(3)
        self.action_space_a3 = Discrete(3)
        self.observation_space = Box(low=np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
                                     high=np.array([[999, 999, 999, 999, 999, 999],
                                                    [999, 999, 999, 999, 999, 999],
                                                    [999, 999, 999, 999, 999, 999]]))
        self.state_agent1 = np.array([0, 0, 0, 0, 0, 0])
        self.state_agent2 = np.array([0, 0, 0, 0, 0, 0])
        self.state_agent3 = np.array([0, 0, 0, 0, 0, 0])
        self.plant_sim = plant_sim

    def step(self,action_a1,action_a2,action_a3):

        # self.plant_sim.executeSimTalk(".Models.Model.RewardAGV1:=0")
        # self.plant_sim.executeSimTalk(".Models.Model.RewardAGV2:=0")
        # self.plant_sim.executeSimTalk(".Models.Model.RewardAGV3:=0")
        self.plant_sim.executeSimTalk(".Models.Model.done:=false")

        if (action_a1 == 0):
            self.plant_sim.executeSimTalk(".Models.Model.moveForward1(.UserObjects.AGV1:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a1 == 1):
            self.plant_sim.executeSimTalk(".Models.Model.moveBackward1(.UserObjects.AGV1:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a1 == 2):
            self.plant_sim.executeSimTalk(".Models.Model.stop1(.UserObjects.AGV1:1)")
            self.plant_sim.startSimulation(".Models.Model")

        if (action_a2 == 0):
            self.plant_sim.executeSimTalk(".Models.Model.moveForward2(.UserObjects.AGV2:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a2 == 1):
            self.plant_sim.executeSimTalk(".Models.Model.moveBackward2(.UserObjects.AGV2:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a2 == 2):
            self.plant_sim.executeSimTalk(".Models.Model.stop2(.UserObjects.AGV2:1)")
            self.plant_sim.startSimulation(".Models.Model")

        if (action_a3 == 0):
            self.plant_sim.executeSimTalk(".Models.Model.moveForward3(.UserObjects.AGV3:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a3 == 1):
            self.plant_sim.executeSimTalk(".Models.Model.moveBackward3(.UserObjects.AGV3:1)")
            self.plant_sim.startSimulation(".Models.Model")
        elif (action_a3 == 2):
            self.plant_sim.executeSimTalk(".Models.Model.stop3(.UserObjects.AGV3:1)")
            self.plant_sim.startSimulation(".Models.Model")

        while True:
            # time.sleep(0.25)

            done = self.plant_sim.getValue(".Models.Model.done")
            if done == True:
                # plant_sim.executeSimTalk(".UserObjects.AGV:1.stopped:=true")
                # plant_sim.executeSimTalk(".UserObjects.AGV:2.stopped:=true")
                # plant_sim.executeSimTalk(".UserObjects.AGV:3.stopped:=true")
                Reward_AGV1 = self.plant_sim.getValue(".Models.Model.RewardAGV1")
                Reward_AGV2 = self.plant_sim.getValue(".Models.Model.RewardAGV2")
                Reward_AGV3 = self.plant_sim.getValue(".Models.Model.RewardAGV3")
                self.plant_sim.executeSimTalk(".Models.Model.AGVPOOL.updateVeh11(.UserObjects.AGV1)")
                self.plant_sim.executeSimTalk(".Models.Model.AGVPOOL.updateVeh21(.UserObjects.AGV2)")
                self.plant_sim.executeSimTalk(".Models.Model.AGVPOOL.updateVeh31(.UserObjects.AGV3)")
                # Reward = [Reward_AGV1,Reward_AGV2,Reward_AGV3]
                global Reward_total
                Reward = [Reward_AGV1, Reward_AGV2, Reward_AGV3]
                Reward_total = self.plant_sim.getValue(".Models.Model.RL")
                self.state = self.Table()
                self.agent1_next_state = self.state[0]
                self.agent2_next_state = self.state[1]
                self.agent3_next_state = self.state[2]

            return self.agent1_next_state, self.agent2_next_state, self.agent3_next_state, Reward_total, done, {}

    def Table(self):

        rows = []
        _rows_coldict = []

        row_count = self.plant_sim.getValue(f".Models.Model.AGVPOOL.final.YDim")
        col_count = self.plant_sim.getValue(f'.Models.Model.AGVPOOL.final.XDim')
        if row_count > 0 and col_count > 0:
            for row_idx in range(row_count + 1):
                row = []
                row_coldict = []
                for col_idx in range(col_count + 1):
                    cell_value = self.plant_sim.getValue(f'.Models.Model.AGVPOOL.final[{col_idx}, {row_idx}]')
                    row.append(cell_value)

                rows.append(row)

            _rows_coldict.append(row_coldict)

            return np.array(rows)

    def reset(self):
        #.plant_sim.resetSimulation(".Models.Model")
        self.state = self.Table()
        self.state_agent1 = self.state[0]
        self.state_agent2 = self.state[1]
        self.state_agent3 = self.state[2]
        return self.state_agent1, self.state_agent2, self.state_agent3

    def render(self):
        self.plant_sim.SetVisible(True)
        self.plant_sim.SetTrustModels(True)

    def close(self):
        self.plant_sim.Quit()


##=============================================================================

##=============================================================================


