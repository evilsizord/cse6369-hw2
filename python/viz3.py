import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# (ref: http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf)
experiments = ['LunarLander_v2_t0', 'LunarLander_v2_t1', 'LunarLander_v2_t2']
for e in range(len(experiments)):
  with open (experiments[e]+'.pkl', 'rb') as f:
    ro_reward = pickle.load(f)
    sns.lineplot(data=ro_reward, linestyle='--', label='t'+str(e))

plt.xlabel('rollout', fontsize=25, labelpad=-2)
plt.ylabel('reward', fontsize=25)
plt.title('Learning curve for LunarLander with AC', fontsize=30)
plt.legend()
plt.grid()
plt.show()