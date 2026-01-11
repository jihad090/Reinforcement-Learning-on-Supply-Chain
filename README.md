# 📦 Supply Chain & Inventory Management Using Reinforcement Learning

This project implements a **Reinforcement Learning (RL)**–based approach to optimize inventory replenishment decisions in a supply chain environment under uncertain demand and lead times.

The goal is to **minimize total operational cost** (holding, stockout, and ordering costs) by learning an adaptive ordering policy using a **Deep Q-Network (DQN)**.

---

## 📌 Project Overview

Traditional inventory control policies (e.g., fixed reorder points) often fail in dynamic and uncertain environments.  
This project models inventory management as a **Markov Decision Process (MDP)** and applies **Deep Reinforcement Learning** to learn optimal restocking decisions from data.

**Key Highlights:**
- Realistic supply chain dataset
- Custom RL environment (Gym-style)
- Deep Q-Network (DQN) agent
- Comparison with a baseline heuristic policy

---

## 🧠 Problem Formulation

The inventory management problem is formulated as an **MDP**:

### 🔹 State (S)
The environment state includes:
- Current inventory level  
- Demand forecast  
- Lead time (days)  
- Promotion flag  
- Truck arrival delay (minutes)

### 🔹 Action (A)
Discrete restocking decisions:
{0, 50, 100, 150, 200} units

csharp
Copy code

### 🔹 Reward (R)
The reward is defined as the **negative total cost**:
Reward = − (Holding Cost + Stockout Cost + Ordering Cost)

css
Copy code

### 🔹 Transition
Inventory evolves according to:
Inventory(t+1) = Inventory(t) + Order − Demand

yaml
Copy code

---

## 📂 Project Structure

supply_chain_rl/
│
├── data/
│ └── Cloud_SupplyChain_Dataset.csv
│
├── env/
│ └── inventory_env.py # Custom RL environment
│
├── agent/
│ └── dqn_agent.py # DQN implementation
│
├── train.py # Training script
├── evaluate.py # Baseline policy evaluation
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 📊 Dataset Description

**Dataset:** Cloud-Based Supply Chain Dataset  

Key features:
- `inventory_level`
- `units_sold`
- `demand_forecast`
- `lead_time_days`
- `restock_quantity`
- `promotion_flag`
- `truck_arrival_delay_mins`

The dataset captures both **operational and logistical uncertainties**, making it suitable for reinforcement learning–based optimization.

---

## ⚙️ Installation & Setup

### 🔹 Prerequisites
- Python 3.9+
- pip / pip3

### 🔹 Install Dependencies
```bash
pip3 install -r requirements.txt
▶️ How to Run the Project
🔹 Train the RL Agent
bash
Copy code
python3 train.py
This will:

Train a DQN agent

Print episode-wise total reward

Display a reward vs episode learning curve

🔹 Evaluate Baseline Policy
bash
Copy code
python3 evaluate.py
This runs a simple heuristic policy (fixed reorder rule) for comparison.

📈 Evaluation Metrics
The performance of the RL agent is evaluated using:

Total operational cost

Episode-wise cumulative reward

Inventory stability

Stockout reduction

The RL policy is compared against a baseline fixed reorder policy.

🧪 Experimental Observations
Initial episodes show poor performance due to exploration

Over time, the agent learns to balance ordering and holding costs

Significant reduction in total cost compared to baseline

Adaptive behavior under demand fluctuations

🛠️ Technologies Used
Python

PyTorch – Deep Q-Network implementation

Pandas / NumPy – Data processing

Matplotlib – Visualization

🚀 Future Improvements
Multi-product inventory optimization

Multi-warehouse (multi-echelon) supply chain

Demand forecasting + RL hybrid model

Continuous action space (DDPG / PPO)

Model deployment for real-time decision making

🎓 Academic Note
This project was developed as part of an AI / Reinforcement Learning course project to demonstrate the application of deep reinforcement learning in real-world supply chain and inventory management problems.

📜 License
This project is for educational and academic use only.

👤 Author
Jihad Hawlader
Department of CSE
