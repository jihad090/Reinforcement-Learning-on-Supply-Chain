from env.inventory_env import InventoryEnv

env = InventoryEnv("data/Cloud_SupplyChain_Dataset.csv")

state = env.reset()
total_cost = 0

while True:
    if state[0] < 50:
        action = 2   # order 100
    else:
        action = 0   # order 0

    state, reward, done = env.step(action)
    total_cost += reward

    if done:
        break

print("Baseline Total Reward:", total_cost)
