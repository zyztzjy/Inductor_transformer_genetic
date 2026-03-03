import numpy as np
import torch
import random
import math


bounds = np.array([
    [1, 5],        # Nt
    [1, 5],        # Nb
    [3.0, 10.0],   # Wt
    [3.0, 10.0],   # Wb
    [40, 120],     # Gc
    [40, 150]      # Di
])

POP_SIZE = 100
N_GEN = 200
ETA_C = 20
ETA_M = 20
P_CROSS = 0.9
P_MUT = 0.1

FSR_MIN = 100.0
AREA_MAX = 200 * 200

LAMBDA0 = 1.0
ALPHA = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_area(ind):
    Nt, Nb, Wt, Wb, Gc, Di = ind
    n = Nt + Nb
    Do = Di + 2 * (Nt*Wt + Nb*Wb + (n-1)*Gc)
    return Do * Do


def evaluate_population(pop, model):

    model.eval()
    with torch.no_grad():
        x = torch.tensor(pop, dtype=torch.float32).to(device)
        y = model(x)

    y = y.cpu().numpy()

    BW  = y[:, 0]
    Q   = y[:, 1]
    IL  = y[:, 2]
    fSR = y[:, 3]

    return BW, Q, IL, fSR

def non_dominated_sort(BW, Q, IL):

    pop_size = len(BW)
    S = [[] for _ in range(pop_size)]
    n = [0] * pop_size
    rank = [0] * pop_size
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):

            if ((BW[p] >= BW[q] and
                 Q[p]  >= Q[q]  and
                 IL[p] <= IL[q]) and
                (BW[p] > BW[q] or
                 Q[p]  > Q[q]  or
                 IL[p] < IL[q])):
                S[p].append(q)

            elif ((BW[q] >= BW[p] and
                   Q[q]  >= Q[p]  and
                   IL[q] <= IL[p]) and
                  (BW[q] > BW[p] or
                   Q[q]  > Q[p]  or
                   IL[q] < IL[p])):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts, rank


def crowding_distance(front, BW, Q, IL):

    distance = np.zeros(len(front))
    objectives = [BW, Q, -IL]

    for obj in objectives:
        values = obj[front]
        sorted_idx = np.argsort(values)

        distance[sorted_idx[0]] = np.inf
        distance[sorted_idx[-1]] = np.inf

        min_val = values[sorted_idx[0]]
        max_val = values[sorted_idx[-1]]

        if max_val - min_val == 0:
            continue

        for i in range(1, len(front)-1):
            distance[sorted_idx[i]] += (
                values[sorted_idx[i+1]] -
                values[sorted_idx[i-1]]
            ) / (max_val - min_val)

    return distance


def sbx(parent1, parent2):

    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(parent1)):
        if random.random() < P_CROSS:

            x1 = parent1[i]
            x2 = parent2[i]
            lb, ub = bounds[i]

            if abs(x1 - x2) > 1e-14:

                x_low = min(x1, x2)
                x_high = max(x1, x2)

                rand = random.random()
                beta = 1.0 + 2.0*(x_low - lb)/(x_high - x_low)
                alpha = 2.0 - beta**-(ETA_C+1)

                if rand <= 1.0/alpha:
                    betaq = (rand*alpha)**(1.0/(ETA_C+1))
                else:
                    betaq = (1.0/(2.0 - rand*alpha))**(1.0/(ETA_C+1))

                child1[i] = 0.5*((x_low + x_high) - betaq*(x_high - x_low))
                child2[i] = 0.5*((x_low + x_high) + betaq*(x_high - x_low))

    return child1, child2


def polynomial_mutation(ind):

    for i in range(len(ind)):
        if random.random() < P_MUT:

            lb, ub = bounds[i]
            delta1 = (ind[i] - lb)/(ub - lb)
            delta2 = (ub - ind[i])/(ub - lb)

            rand = random.random()
            mut_pow = 1.0/(ETA_M+1)

            if rand < 0.5:
                xy = 1 - delta1
                val = 2*rand + (1 - 2*rand)*(xy**(ETA_M+1))
                deltaq = val**mut_pow - 1
            else:
                xy = 1 - delta2
                val = 2*(1 - rand) + 2*(rand - 0.5)*(xy**(ETA_M+1))
                deltaq = 1 - val**mut_pow

            ind[i] += deltaq*(ub - lb)
            ind[i] = np.clip(ind[i], lb, ub)

    return ind


def constraint_violation(pop, fSR):

    cv = np.zeros(len(pop))

    for i, ind in enumerate(pop):
        area = compute_area(ind)

        v1 = max(0, FSR_MIN - fSR[i]) / FSR_MIN
        v2 = max(0, area - AREA_MAX) / AREA_MAX

        cv[i] = v1 + v2

    return cv


def run_nsga2(model):

    pop = np.random.uniform(bounds[:,0], bounds[:,1],
                            (POP_SIZE, len(bounds)))

    for gen in range(N_GEN):

        BW, Q, IL, fSR = evaluate_population(pop, model)
        cv = constraint_violation(pop, fSR)

        lambda_t = LAMBDA0 * math.exp(-ALPHA * gen)

        BW_adj = BW - lambda_t * cv
        Q_adj  = Q  - lambda_t * cv
        IL_adj = IL + lambda_t * cv

        fronts, rank = non_dominated_sort(BW_adj, Q_adj, IL_adj)

        new_pop = []

        for front in fronts:
            if len(new_pop) + len(front) > POP_SIZE:
                distance = crowding_distance(front, BW_adj, Q_adj, IL_adj)
                sorted_idx = np.argsort(-distance)
                for idx in sorted_idx:
                    if len(new_pop) < POP_SIZE:
                        new_pop.append(pop[front[idx]])
                break
            else:
                for idx in front:
                    new_pop.append(pop[idx])

        pop = np.array(new_pop)
        offspring = []
        while len(offspring) < POP_SIZE:
            p1, p2 = random.sample(list(pop), 2)
            c1, c2 = sbx(p1, p2)
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            offspring.append(c1)
            offspring.append(c2)

        pop = np.vstack((pop, offspring[:POP_SIZE]))

        print(f"Generation {gen+1}/{N_GEN}")

    return pop

def get_pareto_front(pop, model):

    BW, Q, IL, fSR = evaluate_population(pop, model)
    fronts, _ = non_dominated_sort(BW, Q, IL)

    idx = fronts[0]

    return pop[idx], BW[idx], Q[idx], IL[idx]
