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
