# **Neuro-Symbolic Multi-Agent System for Intelligent Logistics Routing**

This repository contains the implementation of a **Neuro-Symbolic Multi-Agent System (NS-MAS)** developed for solving logistics optimization tasks, specifically the **Vehicle Routing Problem with Time Windows (VRPTW)**.
The system integrates **neural perception**, **heuristic planning**, and **symbolic reasoning** into a unified architecture that produces explainable and constraint-aware routing decisions.

The project was initially developed offline and later uploaded here for evaluation and further version tracking.

---

# **📌 Overview**

Modern logistics problems require both robust perception and strict constraint satisfaction. This project demonstrates how combining these paradigms yields a hybrid system capable of:

* Interpreting visual inputs using deep learning
* Planning optimal delivery routes under real-world constraints
* Validating solutions using symbolic logic for correctness and explainability

This repository includes the complete implementation up to **Phase B**, covering:

✔ Logistics environment generator
✔ OR-Tools based VRPTW planner
✔ Z3-based symbolic reasoning validator
✔ Synthetic dataset generator for perception
✔ CNN-based Perception Agent (99% accuracy)
✔ End-to-end integration pipeline

---

# **🧠 System Architecture**

The system consists of **three main agents**, each designed to perform a specialized task:

---

## **1. Perception Agent (Neural Network – CNN)**

The perception module predicts customer demand from a synthetic **16×16 grayscale image** that encodes the geometric relationship between a customer and the depot.

### **Key Features**

* Fully synthetic and balanced dataset
* CNN model trained with noise variability
* Achieves **>99% validation accuracy**
* Supports real-time inference inside the pipeline

### **Relevant Files**

* `data/perception_synth.py`
* `train_perception.py`
* `models/perception_net.py`
* `perception_data/` (generated dataset)

---

## **2. Planning Agent (Heuristic Solver – OR-Tools)**

Generates optimized routes for a fleet of vehicles under:

* Capacity constraints
* Time window constraints
* Service time penalties
* Euclidean-distance-based travel times

The planner always produces a feasible solution (when possible) and returns detailed outputs including:

* Route sequences
* Vehicle loads
* Arrival times
* Total objective cost

### **Relevant Files**

* `planner/or_tools_planner.py`

---

## **3. Reasoning Agent (Symbolic Validator – Z3)**

This component validates the planner’s output using **logical constraints**, ensuring:

* Load consistency along each route
* Time monotonicity
* Time-window satisfaction
* Depot arrival consistency

The validator provides a clear **VALID/INVALID** report for each run.

### **Relevant Files**

* `reasoner/z3_validator.py`

---

# **🔄 End-to-End Integration (`run_demo.py`)**

The central script that connects all agents:

1. Generates a logistics instance
2. Solves routing using OR-Tools
3. Predicts customer demands using the CNN
4. Validates the generated routes using Z3
5. Prints human-readable and debug-friendly outputs

### Output Example Includes:

* Customer summary table
* Predicted vs true demand
* Vehicle route breakdown
* Load and time profiles
* Symbolic validation status

---

# **📁 Repository Structure**

```
env/                    Logistics environment generator  
planner/                OR-Tools routing planner  
reasoner/               Z3-based symbolic validator  
data/                   Synthetic dataset generator  
models/                 PerceptionNet and saved models  
perception_data/        Balanced training dataset  
phaseB_make_data.py     Dataset creation script  
train_perception.py     Training pipeline  
run_demo.py             End-to-end system integration  
```

---

# **📊 Results Achieved So Far (Phase B)**

### ✔ Perception Model

* Balanced dataset: 800 samples (200/class)
* Accuracy: **99.3%**
* Stable confusion matrix across all classes
* Successfully integrated inference during routing

### ✔ OR-Tools Planning

* Feasible solutions consistently generated
* Supports 10 customers and 3 vehicles by default
* Diagnostic self-checks for infeasible conditions

### ✔ Symbolic Reasoning

* All OR-Tools solutions validated as **correct**
* Verified constraints: load progression, time consistency, TW constraints

### ✔ Full System

* Neural + heuristic + symbolic pipeline works seamlessly
* Predictions, routes, and validation printed in a unified flow

---

# **🧪 Testing & Validation**

The following tests were conducted:

* **Overfitting Test:** Verified CNN capacity (model fits small batch to 100% accuracy).
* **Dataset Balance Test:** Ensured all classes are equally represented.
* **Symbolic Constraint Tests:** Checked load/time consistency.
* **Route Feasibility Tests:** Executed multiple random test instances.

All modules behave as expected and integrate correctly.

---

# **🚧 Upcoming Work**

The next development stage will extend this project beyond Phase B:

* Reinforcement Learning–based Planning Agent
* Graphical User Interface (GUI) for route visualization
* Advanced evaluation metrics and ablation studies
* Fine-tuning perception noise models
* Visualization of symbolic reasoning steps

---

# **📦 Installation & Usage**

### **1. Install Dependencies**

```
pip install -r requirements.txt
```

### **2. Generate Dataset (Optional)**

```
python phaseB_make_data.py
```

### **3. Train the Perception Model**

```
python train_perception.py
```

### **4. Run the Complete System**

```
python run_demo.py
```

---

# **👨‍💻 Contributors**

* **Yogesh Joshi:** Environment, Planning Agent, Reasoning Agent
* **Yogesh Joshi, Vasu Jain:** Perception Agent, Dataset Pipeline, Model Training
* **Vasu Jain:** Graphical User Interface
  Both members collaborated on integration and testing.

---

# **📄 License**

This project is developed for academic and research purposes.

---

# **⭐ Acknowledgements**

This project uses:

* **PyTorch** for neural model training
* **Google OR-Tools** for routing optimization
* **Z3 SMT Solver** for symbolic reasoning
