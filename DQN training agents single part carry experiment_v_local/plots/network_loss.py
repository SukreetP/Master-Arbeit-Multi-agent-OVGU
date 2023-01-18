import matplotlib.pyplot as plt
import numpy as np


drive_time_a1 = np.loadtxt('loss_history_a1.txt')
drive_time_a2 = np.loadtxt('loss_history_a2.txt')
drive_time_a3 = np.loadtxt('loss_history_a3.txt')
epochs = np.arange(1, 199002)
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
plt.plot(epochs, drive_time_a1,color ='green', label='loss agent 1')
plt.plot(epochs, drive_time_a2,color ='red', label='loss agent 2')
plt.plot(epochs, drive_time_a3,color ='blue', label='loss agent 3')
plt.xlabel('Step')
plt.ylabel('network loss')
#plt.ylim(500,3000)

plt.legend(loc='best')
plt.show()

