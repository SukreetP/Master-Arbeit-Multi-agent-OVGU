import matplotlib.pyplot as plt
import numpy as np


rewards_a1 = np.loadtxt('Rewards_train_a1.txt')
rewards_a2 = np.loadtxt('Rewards_train_a2.txt')
rewards_a3 = np.loadtxt('Rewards_train_a3.txt')
epochs = np.arange(1, 198)
window = 4
avg_reward_1 = []
avg_reward_2 = []
avg_reward_3 = []

for x in range(len(rewards_a1) - window + 1):
    avg_reward_1.append(np.mean(rewards_a1[x:x+window]))

for x in range(len(rewards_a2) - window + 1):
    avg_reward_2.append(np.mean(rewards_a2[x:x+window]))

for x in range(len(rewards_a3) - window + 1):
    avg_reward_3.append(np.mean(rewards_a3[x:x+window]))






#plt.plot(epochs, rewards, label='collision per episode', linewidth = 1)
plt.plot(epochs, avg_reward_1,color ='green', label='rewards per episode agent 1')
plt.plot(epochs, avg_reward_2,color ='red', label='rewards per episode agent 2')
plt.plot(epochs, avg_reward_3,color ='blue', label='rewards per episode agent 3')
plt.xlabel('Epochs')
plt.ylabel('rewards')
#plt.ylim(500,3000)

plt.legend(loc='best')
plt.show()

