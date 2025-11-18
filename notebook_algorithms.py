import random
import numpy as np
import pandas as pd

# Lightweight Node class 
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def child_node(self, action, result_state, cost=0):
        return Node(result_state, parent=self, action=action, path_cost=self.path_cost + cost)


class Problem:
    """Problem wrapper that uses a trained ML model to evaluate states.
    State representation: [water_usage_efficiency, fertilizer_usage, irrigation_frequency]
    """
    def __init__(self, model, features, env_conditions, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max):
        self.model = model
        self.features = features
        self.env = env_conditions.copy()
        self.WUE_min = WUE_min; self.WUE_max = WUE_max
        self.FU_min = FU_min; self.FU_max = FU_max
        self.IF_min = IF_min; self.IF_max = IF_max

    def actions(self, state=None):
        return ["inc_wue","dec_wue","inc_fu","dec_fu","inc_if","dec_if"]

    def result(self, state, action):
        wue, fu, inf = state
        if action == "inc_wue":
            wue = min(self.WUE_max, wue + random.uniform(0.1, (self.WUE_max - self.WUE_min)/6))
        elif action == "dec_wue":
            wue = max(self.WUE_min, wue - random.uniform(0.1, (self.WUE_max - self.WUE_min)/6))
        elif action == "inc_fu":
            fu = min(self.FU_max, fu + random.uniform(1, (self.FU_max - self.FU_min)/6))
        elif action == "dec_fu":
            fu = max(self.FU_min, fu - random.uniform(1, (self.FU_max - self.FU_min)/6))
        elif action == "inc_if":
            inf = min(self.IF_max, inf + 1)
        elif action == "dec_if":
            inf = max(self.IF_min, inf - 1)
        return [wue, fu, int(inf)]

    def value(self, state):
        # Return predicted yield for state
        wue, fu, inf = state
        inp = self.env.copy()
        inp.update({
            'water_usage_efficiency': float(wue),
            'fertilizer_usage': float(fu),
            'irrigation_frequency': int(inf)
        })
        df = pd.DataFrame([{k: inp.get(k, 0) for k in self.features}])
        pred = self.model.predict(df)[0]
        return float(pred)


class GeneticAlgorithm:
    def __init__(self, problem, population_size=30, generations=40, mutation_rate=0.1):
        self.problem = problem
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _random_individual(self):
        wue = random.uniform(self.problem.WUE_min, self.problem.WUE_max)
        fu = random.uniform(self.problem.FU_min, self.problem.FU_max)
        inf = random.randint(self.problem.IF_min, self.problem.IF_max)
        return [wue, fu, inf]

    def _fitness(self, individual):
        return self.problem.value(individual)

    def _crossover(self, a, b):
        # single point crossover for 3-element vector
        return [a[0], b[1], a[2]]

    def _mutate(self, ind):
        if random.random() < self.mutation_rate:
            ind[0] = min(self.problem.WUE_max, max(self.problem.WUE_min, ind[0] + random.uniform(-1,1)))
        if random.random() < self.mutation_rate:
            ind[1] = min(self.problem.FU_max, max(self.problem.FU_min, ind[1] + random.uniform(-5,5)))
        if random.random() < self.mutation_rate:
            ind[2] = min(self.problem.IF_max, max(self.problem.IF_min, ind[2] + random.choice([-1,1])))
        return ind

    def search(self):
        # initialize
        population = [self._random_individual() for _ in range(self.pop_size)]
        for gen in range(self.generations):
            scored = [(self._fitness(ind), ind) for ind in population]
            scored.sort(reverse=True, key=lambda x: x[0])
            # keep top 50%
            cutoff = max(2, len(scored)//2)
            parents = [ind for _, ind in scored[:cutoff]]
            # create new population
            newpop = parents.copy()
            while len(newpop) < self.pop_size:
                a, b = random.sample(parents, 2)
                child = self._crossover(a, b)
                child = self._mutate(child)
                newpop.append(child)
            population = newpop
        # return best
        best = max(population, key=lambda ind: self._fitness(ind))
        return best


class CSP:
    """Simple min-conflicts search for resource allocation."""
    def __init__(self):
        pass

    def min_conflicts(self, model, features, env_conditions, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max, max_steps=1000):
        # initialize random assignment
        current = [random.uniform(WUE_min, WUE_max), random.uniform(FU_min, FU_max), random.randint(IF_min, IF_max)]
        problem = Problem(model, features, env_conditions, WUE_min, WUE_max, FU_min, FU_max, IF_min, IF_max)
        for step in range(max_steps):
            # pick a variable to change at random
            var = random.choice([0,1,2])
            # search for value for var that maximizes predicted yield
            candidates = []
            if var == 0:
                candidates = [random.uniform(WUE_min, WUE_max) for _ in range(10)]
            elif var == 1:
                candidates = [random.uniform(FU_min, FU_max) for _ in range(10)]
            else:
                candidates = [random.randint(IF_min, IF_max) for _ in range(10)]
            best_val = None
            best_score = -1e9
            for val in candidates:
                trial = current.copy()
                trial[var] = val
                score = problem.value(trial)
                if score > best_score:
                    best_score = score
                    best_val = val
            current[var] = best_val
            # yield intermediate
            yield [float(current[0]), float(current[1]), int(current[2])]
        # final
        return [float(current[0]), float(current[1]), int(current[2])]
