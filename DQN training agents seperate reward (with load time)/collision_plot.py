import matplotlib.pyplot as plt
import numpy as np

collision = np.loadtxt('collision_train.txt')
#rewards = np.loadtxt('Rewards.txt')
epochs = np.arange(1, 198)
window = 4
avg_collision = []

for x in range(len(collision) - window + 1):
    avg_collision.append(np.mean(collision[x:x+window]))





#plt.plot(epochs, collision, label='collision per episode', linewidth = 1)
plt.plot(epochs, avg_collision, label='collision per episode')
plt.xlabel('Epochs')
plt.ylabel('number of collision')
#plt.ylim(500,3000)

plt.legend(loc='best')
plt.show()

#plt.plot(epochs, rewards, label='rewards per episode')
#plt.xlabel('Epochs')
#plt.ylabel('rewards')
#plt.ylim(500,3000)

#plt.legend(loc='best')
#plt.show()