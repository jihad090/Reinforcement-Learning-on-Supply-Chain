from env.inventory_env import InventoryEnv
from agent.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

env = InventoryEnv("data/Cloud_SupplyChain_Dataset.csv")
agent = DQNAgent(state_dim=5, action_dim=5)

episodes = 5
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            agent.update_target()
            break

    rewards.append(total_reward)
    print(f"Episode {ep}, Total Reward: {total_reward:.2f}")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Curve")
plt.show()
