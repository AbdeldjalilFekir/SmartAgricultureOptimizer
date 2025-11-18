# 🌾 Smart Agriculture Optimizer

**A Student Project for AI Course 2025**

An AI-powered crop yield optimization system that uses machine learning and optimization algorithms to help maximize agricultural productivity. This project was developed as part of our Artificial Intelligence coursework to explore practical applications of search algorithms and machine learning in agriculture.

## 📋 Project Overview

This project applies various AI techniques learned in class to solve a real-world problem: optimizing crop yields. We implemented and compared four different optimization algorithms to find the best combination of resources (water, fertilizer, irrigation) that maximizes crop production.

**Key Learning Objectives:**
- Implementing search algorithms (A*, Greedy Search)
- Applying evolutionary computation (Genetic Algorithms)
- Solving constraint satisfaction problems (CSP)
- Building machine learning models for prediction
- Creating interactive web interfaces for AI systems

## ✨ What We Built

### 🤖 AI Algorithms Implemented
We implemented and compared 4 different optimization algorithms from our coursework:
- **A* Search**: Uses heuristic function to guide the search
- **Greedy Local Search**: Makes locally optimal choices at each step
- **Genetic Algorithm**: Simulates evolution with selection, crossover, and mutation
- **CSP with Min-Conflicts**: Solves constraint satisfaction problems

### 📊 Web Dashboard
Built a Flask web application that:
- Takes user inputs for crop and environmental conditions
- Runs all 4 algorithms automatically
- Shows comparison results with interactive charts
- Displays which algorithm gives the best yield

### 🔬 Machine Learning Component
- Trained a Linear Regression model on agricultural dataset
- Predicts crop yield based on 13+ parameters (soil nutrients, climate, irrigation, etc.)
- Algorithms optimize the resource allocation to maximize predicted yield

## 🚀 How to Run

### What You Need
- Python 3.11 or higher
- The dataset file (`Crop_recommendation_with_yield.csv`)

### Setup Instructions

1. **Clone or download this repository**

2. **Install required packages**
```bash
pip install -r requirements.txt
```

The main packages we used:
- Flask - for the web interface
- pandas & numpy - for data handling
- scikit-learn - for machine learning
- plotly - for interactive charts

3. **Make sure the dataset is in the project folder**
- The file `Crop_recommendation_with_yield.csv` should be in the same folder as `app.py`

4. **Run the application**
```bash
python app.py
```

5. **Open your browser and go to:**
```
http://127.0.0.1:5000/
```

## 📖 How to Use

### Step 1: Enter Data
Fill in the form with crop and environmental information:
- Soil nutrients (N, P, K)
- Weather conditions (temperature, humidity, rainfall)
- Crop type and soil properties
- Resource constraints (min/max for water, fertilizer, irrigation)

### Step 2: Click "Analyze & Optimize"
The system will:
1. Train the ML model on the dataset
2. Run all 4 AI algorithms automatically
3. Find the best resource combination for maximum yield

### Step 3: See the Results
The results page shows:
- **Baseline Yield**: Expected yield with current/default resources
- **Best Optimized Yield**: Maximum yield found by the algorithms
- **Comparison Table**: Shows what each algorithm recommends
- **Charts**: Visual comparison of algorithm performance and resource usage

## 📁 Project Files

```
SmartAgricultureOptimizer/
│
├── app.py                          # Main Flask application with all algorithms
├── requirements.txt                # List of Python packages needed
├── Crop_recommendation_with_yield.csv  # Dataset for training
├── FINAL_AI_NOTEBOOK_2025.ipynb   # Jupyter notebook with experiments
│
├── templates/
│   ├── index.html                 # Input form page
│   └── results.html               # Results page with charts
│
└── README.md                      # This file
```

## 🧪 About the Algorithms

### 1. A* Search
- Uses a heuristic function to guide the search toward better solutions
- We approximated it with greedy approach for faster performance in web environment

### 2. Greedy Local Search
- Starts with a random solution and keeps making small improvements
- Stops when no better neighboring solution can be found

### 3. Genetic Algorithm
- Inspired by biological evolution
- Creates a population of solutions, combines the best ones, and adds random mutations
- Runs for multiple generations to evolve better solutions

### 4. CSP (Constraint Satisfaction)
- Treats the problem as satisfying constraints (resource limits)
- Uses Min-Conflicts method to randomly assign values while minimizing violations

## 📊 About the Dataset

Our dataset (`Crop_recommendation_with_yield.csv`) contains agricultural data with these features:
- **Soil nutrients**: N, P, K levels
- **Climate data**: temperature, humidity, rainfall, pH
- **Crop info**: crop type (label), growth stage
- **Soil properties**: moisture, soil type, organic matter
- **Resource usage**: water efficiency, fertilizer, irrigation frequency
- **Target**: crop yield in tons per hectare

## 🎓 What We Learned

### Technical Skills
- Implementing different search and optimization algorithms
- Training and using machine learning models
- Building web applications with Flask
- Data visualization with Plotly
- Working with real-world datasets



### Future Improvements
- Add more sophisticated ML models (Random Forest, Neural Networks)
- Implement actual A* with proper path tracking
- Add cost analysis (not just yield maximization)
- Support for multiple crops simultaneously



## 👥 Team Members
FEKIR Abdeldjalil
HADDOUD Mehdi
CHERFIA Abdellah 
BENMAKHLOUF Leryeme 
ACHOUR Djamel Eddine
HENNI Mohemmed Yassine


## 🙏 Acknowledgments

- Dataset source: [Add source if applicable]
- Course: Artificial Intelligence 2025
- Tools used: Python, Flask, scikit-learn, Plotly
- Icons: Font Awesome


**Project completed: May 2025**
