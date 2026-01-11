import numpy as np
import pandas as pd

class InventoryEnv:
    def __init__(self, csv_path, max_steps=500):
        self.data = pd.read_csv(csv_path)
        self.max_steps = max_steps
        self.current_step = 0

        self.holding_cost = 0.1
        self.stockout_cost = 2.0
        self.ordering_cost = 0.5

        self.action_space = [0, 50, 100, 150, 200]

    def reset(self):
        self.current_step = 0
        row = self.data.iloc[self.current_step]
        self.inventory = row["inventory_level"]
        return self._get_state(row)

    def _get_state(self, row):
        return np.array([
            row["inventory_level"],
            row["demand_forecast"],
            row["lead_time_days"],
            row["promotion_flag"],
            row["truck_arrival_delay_mins"]
        ], dtype=np.float32)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        row = self.data.iloc[self.current_step]

        demand = row["units_sold"]

        self.inventory = max(0, self.inventory + action - demand)

        holding = self.inventory * self.holding_cost
        stockout = max(0, demand - self.inventory) * self.stockout_cost
        ordering = action * self.ordering_cost

        reward = -(holding + stockout + ordering)

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        next_row = self.data.iloc[self.current_step]
        next_state = self._get_state(next_row)

        return next_state, reward, done
