import matplotlib.pyplot as plt
import numpy as np


drive_time_a1 = np.loadtxt('wait_a1_time.txt')
drive_time_a2 = np.loadtxt('wait_a2_time.txt')
drive_time_a3 = np.loadtxt('wait_a3_time.txt')
epochs = np.arange(1, 201)
window = 4
avg_reward_1 = []
avg_reward_2 = []
avg_reward_3 = []

for x in range(len(drive_time_a1) - window + 1):
    avg_reward_1.append(np.mean(drive_time_a3[x:x+window]))

for x in range(len(drive_time_a2) - window + 1):
    avg_reward_2.append(np.mean(drive_time_a2[x:x + window]))

for x in range(len(drive_time_a3) - window + 1):
    avg_reward_3.append(np.mean(drive_time_a3[x:x + window]))






#plt.plot(epochs, rewards, label='collision per episode', linewidth = 1)
plt.plot(epochs, drive_time_a1,color ='green', label='wait time per episode agent 1')
plt.plot(epochs, drive_time_a2,color ='red', label='wait time per episode agent 2')
plt.plot(epochs, drive_time_a3,color ='blue', label='wait time per episode agent 3')
plt.xlabel('Episodes')
plt.ylabel('transportation waiting time')
#plt.ylim(500,3000)

plt.legend(loc='best')
plt.show()

