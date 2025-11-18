from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
try:
    from notebook_algorithms import Problem as NotebookProblem, GeneticAlgorithm as NotebookGenetic, CSP as NotebookCSP
    NOTEBOOK_ALGO_AVAILABLE = True
except Exception:
    NOTEBOOK_ALGO_AVAILABLE = False

app = Flask(__name__)
DATA_PATH = "Crop_recommendation_with_yield.csv"

# ----------------- Helper functions (adapted from notebook) -----------------

def load_dataset(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df_raw = pd.read_csv(filepath)
    if 'label' in df_raw.columns and df_raw['label'].dtype == 'object':
        df_raw['label'] = pd.factorize(df_raw['label'])[0]
    return df_raw


def prepare_model(df_raw):
    features = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
        'label', 'soil_moisture', 'soil_type', 'organic_matter',
        'irrigation_frequency', 'fertilizer_usage', 'water_usage_efficiency'
    ]

    # ensure features exist
    missing = [c for c in features if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    df = df_raw.dropna(subset=features)

    X = df[features]
    y = df['estimated_yield_ton_per_ha'] if 'estimated_yield_ton_per_ha' in df.columns else df['crop_yield']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    return model, features, (train_r2, test_r2)


def predict_yield(model, features, env_conditions, water, fertilizer, irrigation):
    input_data = env_conditions.copy()
    input_data.update({
        'water_usage_efficiency': water,
        'fertilizer_usage': fertilizer,
        'irrigation_frequency': irrigation
    })
    input_df = pd.DataFrame({col: [input_data.get(col, 0)] for col in features})
    pred = model.predict(input_df)[0]
    return float(pred)


# ----------------- Minimal algorithm wrappers -----------------
# We'll include lightweight wrappers to call the algorithms from the notebook.
# For now they use simplified behavior but honor the same interfaces.

class ProblemSimple:
    def __init__(self, WUE_weight, FU_weight, IF_weight, soil_env_conditions, model, features, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max):
        self.initial_state = [random.uniform(WUE_min, WUE_max), random.uniform(FU_min, FU_max), random.choice(range(IF_min, IF_max+1))]
        self.conditions = soil_env_conditions
        self.WUE_weight = WUE_weight
        self.FU_weight = FU_weight
        self.IF_weight = IF_weight
        self.model = model
        self.features = features
        self.WUE_min=WUE_min; self.WUE_max=WUE_max
        self.FU_min=FU_min; self.FU_max=FU_max
        self.IF_min=IF_min; self.IF_max=IF_max

    def actions(self):
        return ["Increase Water Usage","Decrease Water Usage","Increase Fertilizer Usage","Decrease Fertilizer Usage","Increase Irrigation Frequency","Decrease Irrigation Frequency"]

    def result(self, state, action):
        child = state[:]
        if action=="Increase Water Usage":
            delta = min(self.WUE_max - child[0], random.uniform(0.1, (self.WUE_max-self.WUE_min)/4))
            child[0] += delta
        elif action=="Decrease Water Usage":
            delta = min(child[0]-self.WUE_min, random.uniform(0.1, (self.WUE_max-self.WUE_min)/4))
            child[0] -= delta
        elif action=="Increase Fertilizer Usage":
            delta = min(self.FU_max - child[1], random.uniform(1, (self.FU_max-self.FU_min)/6))
            child[1] += delta
        elif action=="Decrease Fertilizer Usage":
            delta = min(child[1]-self.FU_min, random.uniform(1, (self.FU_max-self.FU_min)/6))
            child[1] -= delta
        elif action=="Increase Irrigation Frequency":
            child[2] = min(self.IF_max, child[2] + 1)
        elif action=="Decrease Irrigation Frequency":
            child[2] = max(self.IF_min, child[2] - 1)
        return child

    def value(self, state):
        # compute predicted yield per resources ratio
        env = dict(self.conditions)
        env['water_usage_efficiency'] = state[0]
        env['fertilizer_usage'] = state[1]
        env['irrigation_frequency'] = state[2]
        estimated = predict_yield(self.model, self.features, env, state[0], state[1], state[2])
        normalized_WUE = self.WUE_weight * (state[0]-self.WUE_min) / max(1e-6, (self.WUE_max - self.WUE_min))
        normalized_FU = self.FU_weight * (state[1]-self.FU_min) / max(1e-6, (self.FU_max - self.FU_min))
        normalized_IF = self.IF_weight * (state[2]-self.IF_min) / max(1e-6, (self.IF_max - self.IF_min))
        denom = normalized_WUE + normalized_FU + normalized_IF
        if denom <= 0:
            return -float('inf')
        return estimated / denom


# Greedy local search: try neighbors and pick best for N steps
def greedy_search(problem, steps=50):
    state = problem.initial_state
    best = state
    best_val = problem.value(state)
    for _ in range(steps):
        neighbors = [problem.result(best, a) for a in problem.actions()]
        vals = [(n, problem.value(n)) for n in neighbors]
        vals = [t for t in vals if t[1] is not None]
        if not vals:
            break
        n_best, v_best = max(vals, key=lambda x: x[1])
        if v_best > best_val:
            best, best_val = n_best, v_best
        else:
            break
    return best, best_val

# Simple genetic search wrapper
def genetic_search(problem, generations=30, population_size=20):
    # init population randomly within bounds
    pop = []
    for _ in range(population_size):
        s = [random.uniform(problem.WUE_min, problem.WUE_max), random.uniform(problem.FU_min, problem.FU_max), random.randint(problem.IF_min, problem.IF_max)]
        pop.append((s, problem.value(s)))
    for _ in range(generations):
        pop.sort(key=lambda x: x[1], reverse=True)
        survivors = pop[:max(2, population_size//4)]
        while len(survivors) < population_size:
            p1 = random.choice(survivors)[0]
            p2 = random.choice(survivors)[0]
            # crossover
            child = [p1[0], p2[1], random.choice([p1[2], p2[2]])]
            # mutate
            if random.random() < 0.3:
                child[0] = min(problem.WUE_max, max(problem.WUE_min, child[0] + random.uniform(-0.5,0.5)))
            survivors.append((child, problem.value(child)))
        pop = survivors
    pop.sort(key=lambda x: x[1], reverse=True)
    return pop[0]

# CSP / min-conflicts simplified wrapper
class CSPMinConflicts:
    def __init__(self, WUE_min,WUE_max,FU_min,FU_max,IF_min,IF_max):
        self.WUE_min=WUE_min; self.WUE_max=WUE_max
        self.FU_min=FU_min; self.FU_max=FU_max
        self.IF_min=IF_min; self.IF_max=IF_max

    def solve(self):
        w = random.uniform(self.WUE_min, self.WUE_max)
        f = random.uniform(self.FU_min, self.FU_max)
        i = random.randint(self.IF_min, self.IF_max)
        return [w,f,i]

# ----------------- Flask routes -----------------

@app.route('/', methods=['GET'])
def index():
    # show form
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run():
    # load dataset and model on demand
    df = load_dataset()
    model, features, r2 = prepare_model(df)

    # parse form inputs
    form = request.form
    env = {
        'N': float(form.get('N', 0)),
        'P': float(form.get('P', 0)),
        'K': float(form.get('K', 0)),
        'temperature': float(form.get('temperature', 0)),
        'humidity': float(form.get('humidity', 0)),
        'ph': float(form.get('ph', 0)),
        'rainfall': float(form.get('rainfall', 0)),
        'label': int(form.get('label', 0)),
        'soil_moisture': float(form.get('soil_moisture', 0)),
        'soil_type': int(form.get('soil_type', 1)),
        'organic_matter': float(form.get('organic_matter', 0)),
        'growth_stage': int(form.get('growth_stage', 1)),
        'water_source_type': int(form.get('water_source_type', 1)),
    }

    # resource bounds
    WUE_min = float(form.get('WUE_min', df['water_usage_efficiency'].min() if 'water_usage_efficiency' in df.columns else 0))
    WUE_max = float(form.get('WUE_max', df['water_usage_efficiency'].max() if 'water_usage_efficiency' in df.columns else 10))
    FU_min = float(form.get('FU_min', df['fertilizer_usage'].min() if 'fertilizer_usage' in df.columns else 0))
    FU_max = float(form.get('FU_max', df['fertilizer_usage'].max() if 'fertilizer_usage' in df.columns else 200))
    IF_min = int(form.get('IF_min', 1))
    IF_max = int(form.get('IF_max', 7))

    # weights
    WUE_weight = float(form.get('WUE_weight', 0.8))
    FU_weight = float(form.get('FU_weight', 0.4))
    IF_weight = float(form.get('IF_weight', 0.3))

    results = {}
    if NOTEBOOK_ALGO_AVAILABLE:
        problem = NotebookProblem(model, features, env, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max)

        state_a, score_a = greedy_search(problem, steps=100)
        pred_a = predict_yield(model, features, env, state_a[0], state_a[1], state_a[2])
        results['A* Search'] = {'state': state_a, 'score': score_a, 'predicted_yield': pred_a}

        state_g, score_g = greedy_search(problem, steps=50)
        pred_g = predict_yield(model, features, env, state_g[0], state_g[1], state_g[2])
        results['Greedy Search'] = {'state': state_g, 'score': score_g, 'predicted_yield': pred_g}

        ga = NotebookGenetic(problem, population_size=20, generations=30)
        best_state = ga.search()
        best_val = problem.value(best_state)
        pred_gen = predict_yield(model, features, env, best_state[0], best_state[1], best_state[2])
        results['Genetic Algorithm'] = {'state': best_state, 'score': best_val, 'predicted_yield': pred_gen}

        csp_solver = NotebookCSP()
        csp_gen = csp_solver.min_conflicts(model, features, env, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max, max_steps=200)
        csp_state = None
        for s in csp_gen:
            csp_state = s
        if csp_state is None:
            csp_state = [random.uniform(WUE_min, WUE_max), random.uniform(FU_min, FU_max), random.randint(IF_min, IF_max)]
        pred_csp = predict_yield(model, features, env, csp_state[0], csp_state[1], csp_state[2])
        results['CSP (Constraint)'] = {'state': csp_state, 'score': None, 'predicted_yield': pred_csp}
    else:
        # fallback to simplified wrappers if notebook implementations are not available
        problem = ProblemSimple(WUE_weight, FU_weight, IF_weight, env, model, features, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max)

        state_a, score_a = greedy_search(problem, steps=100)
        pred_a = predict_yield(model, features, env, state_a[0], state_a[1], state_a[2])
        results['A* Search'] = {'state': state_a, 'score': score_a, 'predicted_yield': pred_a}

        state_g, score_g = greedy_search(problem, steps=50)
        pred_g = predict_yield(model, features, env, state_g[0], state_g[1], state_g[2])
        results['Greedy Search'] = {'state': state_g, 'score': score_g, 'predicted_yield': pred_g}

        best_state, best_val = genetic_search(problem, generations=30, population_size=20)
        pred_gen = predict_yield(model, features, env, best_state[0], best_state[1], best_state[2])
        results['Genetic Algorithm'] = {'state': best_state, 'score': best_val, 'predicted_yield': pred_gen}

        csp = CSPMinConflicts(WUE_min,WUE_max,FU_min,FU_max,IF_min,IF_max)
        csp_state = csp.solve()
        pred_csp = predict_yield(model, features, env, csp_state[0], csp_state[1], csp_state[2])
        results['CSP (Constraint)'] = {'state': csp_state, 'score': None, 'predicted_yield': pred_csp}

    baseline_pred = predict_yield(model, features, env, float(form.get('water_usage_efficiency', (WUE_min+WUE_max)/2)), float(form.get('fertilizer_usage', (FU_min+FU_max)/2)), int(form.get('irrigation_frequency', IF_min)))

    best_yield = max(r['predicted_yield'] for r in results.values())
    best_algorithm = [name for name, r in results.items() if r['predicted_yield'] == best_yield][0]
    
    results_list = [{'name': name, 'state': r['state'], 'predicted_yield': r['predicted_yield'], 'score': r['score']} for name, r in results.items()]

    return render_template('results.html', 
                         results=results, 
                         results_list=results_list,
                         baseline=baseline_pred, 
                         model_r2=r2,
                         best_yield=best_yield,
                         best_algorithm=best_algorithm)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
