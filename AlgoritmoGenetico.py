from typing import Dict, List, Tuple, Optional
import math
from collections import Counter
import random

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Datos del problema

species = ["AL", "AS", "ASC", "AST", "OC", "OE", "OR", "OS", "PL", "YF"]

species_names = {
    "AL": "Agave lechuguilla",
    "AS": "Agave salmiana",
    "ASC": "Agave scabra",
    "AST": "Agave striata",
    "OC": "Opuntia cantabrigiensis",
    "OE": "Opuntia engelmannii",
    "OR": "Opuntia robusta",
    "OS": "Opuntia streptacantha",
    "PL": "Prosopis laevigata",
    "YF": "Yucca filifera",
}

#  Coeficientes de Compatibilidad
theta_values = {
    "AL": [-0.05, -0.05, -0.05, -0.05, +0.05, +0.05, +0.05, +0.05, +0.20, +0.05],
    "AS": [-0.05, -0.05, -0.05, -0.05, +0.05, +0.05, +0.05, +0.05, +0.20, +0.05],
    "ASC": [-0.05, -0.05, -0.05, -0.05, +0.05, +0.05, +0.05, +0.05, +0.20, +0.05],
    "AST": [-0.05, -0.05, -0.05, -0.05, +0.05, +0.05, +0.05, +0.05, +0.20, +0.05],
    "OC":  [+0.05, +0.05, +0.05, +0.05, -0.05, -0.05, -0.05, -0.05, +0.25, +0.05],
    "OE":  [+0.05, +0.05, +0.05, +0.05, -0.05, -0.05, -0.05, -0.05, +0.25, +0.05],
    "OR":  [+0.05, +0.05, +0.05, +0.05, -0.05, -0.05, -0.05, -0.05, +0.25, +0.05],
    "OS":  [+0.05, +0.05, +0.05, +0.05, -0.05, -0.05, -0.05, -0.05, +0.25, +0.05],
    "PL":  [+0.05, +0.05, +0.05, +0.05, +0.05, +0.05, +0.05, +0.05,  0.00, +0.05],
    "YF":  [+0.05, +0.05, +0.05, +0.05, +0.05, +0.05, +0.05, +0.05, +0.15,  0.00],
}

theta: Dict[Tuple[str, str], float] = {}
for neighbor_sp, row_vals in theta_values.items():
    for j, receptor_sp in enumerate(species):
        theta[(neighbor_sp, receptor_sp)] = row_vals[j]

# Densidad por hectarea
density_per_ha = {
    "AL": 42,
    "AS": 196,
    "ASC": 42,
    "AST": 42,
    "OC": 49,
    "OE": 38,
    "OR": 73,
    "OS": 64,
    "PL": 86,
    "YF": 26,
}
total_density = 658

# Areas de los poligonos
polygon_areas = {
    "P1": 1.28,
    "P2": 6.64,
    "P3": 6.76,
    "P4": 1.38,
    "P5": 8.00,
    "P6": 7.82,
    "P7": 5.53,
    "P8": 5.64,
    "P9": 7.11,
    "P10": 6.11,
    "P11": 5.64,
    "P12": 4.92,
    "P13": 5.05,
    "P14": 4.75,
    "P15": 7.97,
    "P16": 7.34,
    "P17": 5.98,
    "P18": 5.40,
    "P19": 6.28,
    "P20": 7.60,
    "P21": 8.00,
    "P22": 8.00,
    "P23": 7.67,
    "P24": 1.47,
    "P25": 4.19,
    "P26": 7.52,
    "P27": 8.00,
    "P28": 8.00,
    "P29": 7.56,
    "P30": 5.40,
}

# Plantas existentes
existing_counts_per_polygon: Dict[str, Dict[str, int]] = {
    "P1": {"AL": 8, "AS": 46, "ASC": 16, "AST": 16, "OC": 11, "OE": 14, "OR": 18, "OS": 12, "PL": 15, "YF": 9},
    "P2": {"AL": 58, "AS": 263, "ASC": 47, "AST": 49, "OC": 60, "OE": 44, "OR": 111, "OS": 95, "PL": 98, "YF": 39},
    "P3": {"AL": 66, "AS": 236, "ASC": 51, "AST": 50, "OC": 71, "OE": 56, "OR": 100, "OS": 78, "PL": 123, "YF": 28},
    "P4": {"AL": 10, "AS": 52, "ASC": 15, "AST": 9, "OC": 15, "OE": 10, "OR": 18, "OS": 20, "PL": 23, "YF": 11},
    "P5": {"AL": 65, "AS": 280, "ASC": 66, "AST": 61, "OC": 92, "OE": 54, "OR": 124, "OS": 114, "PL": 106, "YF": 41},
    "P6": {"AL": 67, "AS": 306, "ASC": 63, "AST": 61, "OC": 81, "OE": 62, "OR": 118, "OS": 91, "PL": 133, "YF": 45},
    "P7": {"AL": 41, "AS": 209, "ASC": 43, "AST": 49, "OC": 48, "OE": 52, "OR": 87, "OS": 60, "PL": 97, "YF": 27},
    "P8": {"AL": 32, "AS": 252, "ASC": 46, "AST": 40, "OC": 73, "OE": 39, "OR": 86, "OS": 79, "PL": 91, "YF": 26},
    "P9": {"AL": 58, "AS": 269, "ASC": 58, "AST": 53, "OC": 57, "OE": 58, "OR": 94, "OS": 91, "PL": 117, "YF": 37},
    "P10": {"AL": 47, "AS": 209, "ASC": 47, "AST": 41, "OC": 66, "OE": 43, "OR": 96, "OS": 71, "PL": 108,"YF": 29},
    "P11": {"AL": 48, "AS": 233, "ASC": 41, "AST": 37, "OC": 48, "OE": 40, "OR": 94, "OS": 65, "PL": 94, "YF": 27},
    "P12": {"AL": 36, "AS": 182, "ASC": 43, "AST": 44, "OC": 51, "OE": 41, "OR": 68, "OS": 72, "PL": 67, "YF": 19},
    "P13": {"AL": 50, "AS": 189, "ASC": 49, "AST": 28, "OC": 43, "OE": 33, "OR": 73, "OS": 61, "PL": 97, "YF": 25},
    "P14": {"AL": 35, "AS": 186, "ASC": 38, "AST": 38, "OC": 49, "OE": 44, "OR": 96, "OS": 89, "PL": 132,"YF": 23},
    "P15": {"AL": 40, "AS": 311, "ASC": 52, "AST": 52, "OC": 68, "OE": 50, "OR": 132,"OS": 77, "PL": 149,"YF": 53},
    "P16": {"AL": 70, "AS": 290, "ASC": 55, "AST": 54, "OC": 84, "OE": 45, "OR": 105,"OS": 81, "PL": 121,"YF": 40},
    "P17": {"AL": 56, "AS": 233, "ASC": 39, "AST": 46, "OC": 71, "OE": 44, "OR": 83, "OS": 82, "PL": 97, "YF": 32},
    "P18": {"AL": 45, "AS": 204, "ASC": 34, "AST": 33, "OC": 56, "OE": 46, "OR": 92, "OS": 90, "PL": 120,"YF": 17},
    "P19": {"AL": 56, "AS": 223, "ASC": 51, "AST": 48, "OC": 77, "OE": 48, "OR": 93, "OS": 107,"PL": 104,"YF": 33},
    "P20": {"AL": 69, "AS": 285, "ASC": 57, "AST": 75, "OC": 89, "OE": 54, "OR": 107,"OS": 103,"PL": 139,"YF": 25},
    "P21": {"AL": 54, "AS": 287, "ASC": 70, "AST": 62, "OC": 89, "OE": 77, "OR": 126,"OS": 109,"PL": 139,"YF": 44},
    "P22": {"AL": 75, "AS": 292, "ASC": 60, "AST": 68, "OC": 104,"OE": 109,"OR": 125,"OS": 95, "PL": 125,"YF": 43},
    "P23": {"AL": 44, "AS": 288, "ASC": 69, "AST": 69, "OC": 69, "OE": 122,"OR": 100,"OS": 106,"PL": 130,"YF": 39},
    "P24": {"AL": 10, "AS": 59,  "ASC": 15, "AST": 11, "OC": 17, "OE": 17, "OR": 17, "OS": 39, "PL": 39, "YF": 8},
    "P25": {"AL": 44, "AS": 155, "ASC": 28, "AST": 31, "OC": 39, "OE": 65, "OR": 50, "OS": 67, "PL": 122,"YF": 25},
    "P26": {"AL": 72, "AS": 291, "ASC": 73, "AST": 55, "OC": 66, "OE": 125,"OR": 100,"OS": 122,"PL": 129,"YF": 31},
    "P27": {"AL": 74, "AS": 342, "ASC": 52, "AST": 66, "OC": 70, "OE": 100,"OR": 100,"OS": 129,"PL": 141,"YF": 44},
    "P28": {"AL": 67, "AS": 309, "ASC": 61, "AST": 53, "OC": 86, "OE": 120,"OR": 103,"OS": 141,"PL": 135,"YF": 33},
    "P29": {"AL": 60, "AS": 280, "ASC": 56, "AST": 55, "OC": 76, "OE": 76, "OR": 85, "OS": 86, "PL": 86, "YF": 36},
    "P30": {"AL": 34, "AS": 195, "ASC": 56, "AST": 58, "OC": 55, "OE": 63, "OR": 63, "OS": 83, "PL": 63, "YF": 23}
}

# Funciones base 

def build_grid_neighbors(rows: int, cols: int) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {}
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            neigh = []
            if r > 0:
                neigh.append((r - 1) * cols + c)      # up
            if r < rows - 1:
                neigh.append((r + 1) * cols + c)      # down
            if c > 0:
                neigh.append(r * cols + (c - 1))      # left
            if c < cols - 1:
                neigh.append(r * cols + (c + 1))      # right
            neighbors[node] = neigh
    return neighbors


def greedy_assignment(
    species_list: List[str],
    nodes: List[int],
    neighbors: Dict[int, List[int]],
    theta_dict: Dict[Tuple[str, str], float],
    demand_new: Dict[str, int],
    capacity: int,
    existing_assignment: Optional[Dict[int, str]] = None,
):
    """
    Heurística codiciosa para asignar especies a nodos en un polígono.
    """
    if existing_assignment is None:
        existing_assignment = {}

    assignment: Dict[int, str] = {}
    plants_per_species: Dict[str, int] = {s: 0 for s in species_list}
    total_plants = 0

    for n, sp in existing_assignment.items():
        assignment[n] = sp
        plants_per_species[sp] = plants_per_species.get(sp, 0) + 1
        total_plants += 1

    # Demanda restante
    remaining_demand: Dict[str, int] = {
        s: max(demand_new.get(s, 0), 0)
        for s in species_list
    }

    # Ordenar los nodos sin asignar por número de vecinos (los más conectados primero)
    unassigned_nodes = [n for n in nodes if n not in assignment]
    unassigned_nodes.sort(
        key=lambda n: len(neighbors.get(n, [])),
        reverse=True,
    )

    def local_score(sp: str, node: int) -> float:
        """
        Puntuación de compatibilidad si colocamos la especie sp en este nodo.
        """
        score = 0.0
        for nb in neighbors.get(node, []):
            if nb in assignment:
                sp_nb = assignment[nb]
                score += theta_dict.get((sp_nb, sp), 0.0)
        return score

    for n in unassigned_nodes:
        if total_plants >= capacity:
            break

        candidates = [s for s in species_list if remaining_demand.get(s, 0) > 0]

        if not candidates:
            candidates = list(species_list)

        best_species = None
        best_score = float("-inf")

        for s in candidates:
            sc = local_score(s, n)
            if (
                sc > best_score
                or (
                    sc == best_score
                    and best_species is not None
                    and remaining_demand.get(s, 0)
                    > remaining_demand.get(best_species, 0)
                )
            ):
                best_score = sc
                best_species = s

        if best_species is None:
            continue

        assignment[n] = best_species
        plants_per_species[best_species] += 1
        total_plants += 1
        if remaining_demand.get(best_species, 0) > 0:
            remaining_demand[best_species] -= 1

    # Valor objetivo de compatibilidad
    objective = 0.0
    seen_edges = set()
    for n in nodes:
        for nb in neighbors.get(n, []):
            if (nb, n) in seen_edges or n == nb:
                continue
            if n in assignment and nb in assignment:
                s1 = assignment[n]
                s2 = assignment[nb]
                objective += theta_dict.get((s1, s2), 0.0)
            seen_edges.add((n, nb))

    return assignment, objective


def create_existing_assignment(
    nodes: List[int],
    existing_counts: Dict[str, int],
    seed: int = 42,
) -> Dict[int, str]:
    """
    Coloca aleatoriamente las plantas ya existentes en los nodos de la cuadrícula.
    """
    rng = random.Random(seed)
    free_nodes = nodes.copy()
    rng.shuffle(free_nodes)

    assignment: Dict[int, str] = {}
    idx = 0
    for sp, count in existing_counts.items():
        for _ in range(count):
            if idx >= len(free_nodes):
                break
            node = free_nodes[idx]
            assignment[node] = sp
            idx += 1
    return assignment


def evaluate_fitness(
    genes: List[str],
    species_list: List[str],
    neighbors: Dict[int, List[int]],
    theta_dict: Dict[Tuple[str, str], float],
    target_counts: Dict[str, int],
    alpha: float = 5.0,
) -> float:
    """
    Fitness = compatibility_sum - alpha * demand_penalty.
    demand_penalty = sum of absolute deviations from target_counts.
    """
    counts = {s: 0 for s in species_list}
    for sp in genes:
        if sp in counts:
            counts[sp] += 1

    # Compatibilidad
    comp_sum = 0.0
    seen_edges = set()
    num_nodes = len(genes)
    for n in range(num_nodes):
        for nb in neighbors.get(n, []):
            if nb < 0 or nb >= num_nodes:
                continue
            if (nb, n) in seen_edges or n == nb:
                continue
            s1 = genes[n]
            s2 = genes[nb]
            comp_sum += theta_dict.get((s1, s2), 0.0)
            seen_edges.add((n, nb))

    # Penalización por demanda
    penalty = 0.0
    for s in species_list:
        diff = counts.get(s, 0) - target_counts.get(s, 0)
        penalty += abs(diff)

    fitness = comp_sum - alpha * penalty
    return fitness


def random_individual(
    num_nodes: int,
    species_list: List[str],
    fixed_species_by_node: Dict[int, str],
    target_counts: Dict[str, int],
    rng: random.Random,
) -> List[str]:
    """
    Generar un individuo aleatorio respetando los nodos fijos y
    aproximando la distribución target_counts.
    """
    genes = [None] * num_nodes

    # Genes fijos
    for n, sp in fixed_species_by_node.items():
        if 0 <= n < num_nodes:
            genes[n] = sp

    free_positions = [i for i in range(num_nodes) if genes[i] is None]

    total_target = sum(target_counts.values())
    if total_target == 0:
        probs = [1.0 / len(species_list)] * len(species_list)
    else:
        probs = [target_counts[s] / total_target for s in species_list]

    for pos in free_positions:
        r = rng.random()
        cum = 0.0
        for s, p in zip(species_list, probs):
            cum += p
            if r <= cum:
                genes[pos] = s
                break
        if genes[pos] is None:
            genes[pos] = rng.choice(species_list)

    return genes


def greedy_individual_from_scratch(
    nodes: List[int],
    neighbors: Dict[int, List[int]],
    theta_dict: Dict[Tuple[str, str], float],
    species_list: List[str],
    total_demand: Dict[str, int],
    capacity: int,
    existing_assignment: Optional[Dict[int, str]],
    rng: random.Random,
) -> List[str]:
    """
    Se usa la heurística codiciosa para construir un buen individuo inicial.
    """
    num_nodes = len(nodes)

    # calcular la nueva demanda si tenemos plantas existentes
    if existing_assignment is not None:
        existing_counts = {s: 0 for s in species_list}
        for sp in existing_assignment.values():
            if sp in existing_counts:
                existing_counts[sp] += 1
        new_demand = {
            s: max(total_demand[s] - existing_counts.get(s, 0), 0)
            for s in species_list
        }
    else:
        new_demand = total_demand

    greedy_assign, _ = greedy_assignment(
        species_list=species_list,
        nodes=nodes,
        neighbors=neighbors,
        theta_dict=theta,
        demand_new=new_demand,
        capacity=capacity,
        existing_assignment=existing_assignment,
    )

    genes = [None] * num_nodes
    for n in nodes:
        if n in greedy_assign:
            genes[n] = greedy_assign[n]

    # Se completan los genes que faltan al azar
    for i in range(num_nodes):
        if genes[i] is None:
            genes[i] = rng.choice(species_list)

    return genes


def crossover(
    parent1: List[str],
    parent2: List[str],
    fixed_mask: List[bool],
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """
    Cruce de un punto con genes fijos restaurados
    """
    num = len(parent1)
    if num < 2:
        return parent1[:], parent2[:]
    cut = rng.randint(1, num - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]

    for i, is_fixed in enumerate(fixed_mask):
        if is_fixed:
            child1[i] = parent1[i]
            child2[i] = parent1[i]

    return child1, child2


def mutate(
    genes: List[str],
    species_list: List[str],
    fixed_mask: List[bool],
    pm: float,
    rng: random.Random,
) -> None:
    """
    Mutación de genes no fijos con probabilidad pm.
    """
    for i in range(len(genes)):
        if fixed_mask[i]:
            continue
        if rng.random() < pm:
            current = genes[i]
            choices = [s for s in species_list if s != current]
            if choices:
                genes[i] = rng.choice(choices)


def evaluate_solution(
    genes: List[str],
    polygon_id: str,
    neighbors: Dict[int, List[int]]
):
    """
    Calcula:
      - compatibilidad total (sumatoria theta en aristas)
      - desviación de demanda (|asignado - objetivo|)
    """
    area = polygon_areas[polygon_id]
    objetivo = {s: round(density_per_ha[s] * area) for s in species}

    # compatibilidad
    comp_sum = 0.0
    seen = set()
    num_nodes = len(genes)
    for n in range(num_nodes):
        for nb in neighbors.get(n, []):
            if nb < 0 or nb >= num_nodes:
                continue
            if (nb, n) in seen or n == nb:
                continue
            s1 = genes[n]
            s2 = genes[nb]
            comp_sum += theta.get((s1, s2), 0.0)
            seen.add((n, nb))

    # desviación
    counts = Counter(genes)
    dev = sum(abs(counts.get(s, 0) - objetivo[s]) for s in species)

    return comp_sum, dev


# Simulación Monte Carlo 

def simulate_one_hectare():
    """
    Simula un escenario de 1 ha:
    - Genera plantas existentes por especie (Poisson con media = densidad por ha).
    - Coloca las plantas aleatoriamente en una malla de tamaño aprox. total_density.
    - Calcula:
        total_plants
        existing_counts
        purchases
        comp_sum
        transition_counts
    """
    # Número de plantas existentes por especie (aleatorio Poisson)
    existing_counts = {
        s: np.random.poisson(lam=density_per_ha[s])
        for s in species
    }
    total_plants = sum(existing_counts.values())

    # Metas (objetivo) por especie en 1 ha
    target_counts = {
        s: density_per_ha[s]
        for s in species
    }

    # Plantas a suministrar = max(objetivo - existentes, 0)
    purchases = {
        s: max(target_counts[s] - existing_counts[s], 0)
        for s in species
    }

    # Construir grid para 1 ha (usando total_density)
    side = int(math.ceil(math.sqrt(total_density)))
    rows, cols = side, side
    num_cells = rows * cols
    neighbors = build_grid_neighbors(rows, cols)

    # Colocar plantas existentes aleatoriamente en la malla
    all_positions = np.arange(num_cells)
    np.random.shuffle(all_positions)

    assignment = [None] * num_cells
    idx = 0
    for s in species:
        for _ in range(existing_counts[s]):
            if idx >= num_cells:
                break
            pos = all_positions[idx]
            assignment[pos] = s
            idx += 1

    # Calcular compatibilidad total y transiciones
    comp_sum = 0.0
    transition_counts = {si: Counter() for si in species}

    seen_edges = set()
    for n in range(num_cells):
        si = assignment[n]
        if si is None:
            continue

        for nb in neighbors.get(n, []):
            if nb < 0 or nb >= num_cells:
                continue
            if (nb, n) in seen_edges or n == nb:
                continue

            sj = assignment[nb]
            if sj is None:
                continue

            # Compatibilidad (theta(si, sj))
            comp_sum += theta.get((si, sj), 0.0)
            # Contar transición en ambas direcciones
            transition_counts[si][sj] += 1
            transition_counts[sj][si] += 1

            seen_edges.add((n, nb))

    return total_plants, existing_counts, purchases, comp_sum, transition_counts


def run_monte_carlo_hectarea(n_sim: int = 1000):
    """
    Corre n_sim simulaciones de 1 ha y regresa:
      - expected_total_plants
      - expected_existing_per_species
      - expected_purchases_per_species
      - expected_competition
      - transition_prob
      - planting_order
    """
    total_plants_samples = []
    comp_samples = []

    sum_existing_per_species = {s: 0.0 for s in species}
    sum_purchases_per_species = {s: 0.0 for s in species}

    global_transition_counts = {si: Counter() for si in species}

    for _ in range(n_sim):
        total_plants, existing_counts, purchases, comp_sum, trans_counts = simulate_one_hectare()

        total_plants_samples.append(total_plants)
        comp_samples.append(comp_sum)

        for s in species:
            sum_existing_per_species[s] += existing_counts[s]
            sum_purchases_per_species[s] += purchases[s]

        for si in species:
            global_transition_counts[si].update(trans_counts[si])

    expected_total_plants = float(np.mean(total_plants_samples))

    expected_existing_per_species = {
        s: sum_existing_per_species[s] / n_sim
        for s in species
    }

    expected_purchases_per_species = {
        s: sum_purchases_per_species[s] / n_sim
        for s in species
    }

    expected_competition = float(np.mean(comp_samples))

    transition_prob = {}
    for si in species:
        row = global_transition_counts[si]
        total_trans = sum(row.values())
        if total_trans > 0:
            transition_prob[si] = {
                sj: row[sj] / total_trans for sj in species
            }
        else:
            transition_prob[si] = {sj: 0.0 for sj in species}

    row_effect = {
        s: sum(theta.get((s, t), 0.0) for t in species)
        for s in species
    }
    planting_order = sorted(species, key=lambda s: row_effect[s], reverse=True)

    results = {
        "expected_total_plants": expected_total_plants,
        "expected_existing_per_species": expected_existing_per_species,
        "expected_purchases_per_species": expected_purchases_per_species,
        "expected_competition": expected_competition,
        "transition_prob": transition_prob,
        "planting_order": planting_order,
    }
    return results


def genetic_algorithm_polygon(
    polygon_id: str,
    pop_size: int = 60,
    generations: int = 50,
    pc: float = 0.8,
    pm: float = 0.01,
    use_existing: bool = True,
    seed: int = 123,
    with_simulation: bool = False,
    n_sim: int = 1000,
):
    """
    Algoritmo Genético para un polígono.
    Si with_simulation=True, al final ejecuta la simulación Monte Carlo (1 ha)
    y regresa también sim_results.
    """
    rng = random.Random(seed)

    if polygon_id not in polygon_areas:
        raise ValueError(f"Polígono desconocido: {polygon_id}")

    area = polygon_areas[polygon_id]

    target_counts: Dict[str, int] = {
        s: round(density_per_ha[s] * area)
        for s in species
    }

    capacity = round(total_density * area)
    side = math.ceil(math.sqrt(capacity))
    rows, cols = side, side
    num_nodes = rows * cols
    nodes = list(range(num_nodes))
    neighbors = build_grid_neighbors(rows, cols)

    existing_counts = existing_counts_per_polygon.get(polygon_id, None)
    if use_existing and existing_counts is not None:
        existing_assignment = create_existing_assignment(nodes, existing_counts, seed=seed)
    else:
        existing_assignment = None

    fixed_species_by_node: Dict[int, str] = {}
    fixed_mask = [False] * num_nodes
    if existing_assignment is not None:
        for n, sp in existing_assignment.items():
            fixed_species_by_node[n] = sp
            if 0 <= n < num_nodes:
                fixed_mask[n] = True

    population: List[List[str]] = []

    elite = greedy_individual_from_scratch(
        nodes=nodes,
        neighbors=neighbors,
        theta_dict=theta,
        species_list=species,
        total_demand=target_counts,
        capacity=capacity,
        existing_assignment=existing_assignment,
        rng=rng,
    )
    population.append(elite)

    while len(population) < pop_size:
        indiv = random_individual(
            num_nodes=num_nodes,
            species_list=species,
            fixed_species_by_node=fixed_species_by_node,
            target_counts=target_counts,
            rng=rng,
        )
        population.append(indiv)

    best_individual = None
    best_fitness = float("-inf")  # fitness ya desplazado (positivo)
    best_history = []

    for _gen in range(generations):
        raw_fitness_vals = []

        # 1) Calculamos fitness crudo para toda la población
        for indiv in population:
            raw_fit = evaluate_fitness(
                genes=indiv,
                species_list=species,
                neighbors=neighbors,
                theta_dict=theta,
                target_counts=target_counts,
            )
            raw_fitness_vals.append(raw_fit)

        # 2) Desplazamos para que el mínimo sea 1
        min_raw = min(raw_fitness_vals)
        shift = -min_raw + 1 if min_raw <= 0 else 0.0

        fitness_vals = [rf + shift for rf in raw_fitness_vals]

        # 3) Actualizamos mejor individuo con el fitness ya desplazado
        for indiv, fit in zip(population, fitness_vals):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = indiv.copy()

        best_history.append(best_fitness)

        # 4) Elitismo y reproducción usando fitness_vals (ya positivos)
        new_pop: List[List[str]] = [best_individual.copy()]

        while len(new_pop) < pop_size:
            i1, i2 = rng.randrange(pop_size), rng.randrange(pop_size)
            parent1 = population[i1] if fitness_vals[i1] > fitness_vals[i2] else population[i2]

            j1, j2 = rng.randrange(pop_size), rng.randrange(pop_size)
            parent2 = population[j1] if fitness_vals[j1] > fitness_vals[j2] else population[j2]

            if rng.random() < pc:
                child1, child2 = crossover(parent1, parent2, fixed_mask, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            mutate(child1, species, fixed_mask, pm, rng)
            mutate(child2, species, fixed_mask, pm, rng)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        population = new_pop

    comp_sum, dev = evaluate_solution(best_individual, polygon_id, neighbors)

    sim_results = None
    if with_simulation:
        sim_results = run_monte_carlo_hectarea(n_sim=n_sim)

    return best_individual, best_fitness, best_history, (rows, cols), comp_sum, dev, sim_results


def compute_targets_and_compras(polygon_id: str):
    """
    Para un polígono p, calcula:
      - objetivo_{s,p}
      - existentes_{s,p}
      - compras_{s,p}
    """
    area = polygon_areas[polygon_id]
    objetivo = {s: round(density_per_ha[s] * area) for s in species}
    existentes = existing_counts_per_polygon.get(polygon_id, {s: 0 for s in species})
    compras = {
        s: max(objetivo[s] - existentes.get(s, 0), 0)
        for s in species
    }
    return objetivo, existentes, compras


def plot_assignment(genes: List[str], rows: int, cols: int, title: str):
    """
    Crea la figura de la cuadrícula de asignación
    """
    grid = np.array(genes).reshape(rows, cols)

    unique_species = sorted(set(genes))
    code = {sp: i for i, sp in enumerate(unique_species)}
    grid_num = np.vectorize(code.get)(grid)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid_num, interpolation="nearest", cmap="Greens")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="species code")
    fig.tight_layout()
    return fig


# Interfaz Streamlit

st.set_page_config(page_title="Reforestación por Polígonos", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3, h4 {
        color: #145A32; /* verde bosque */
    }
    .stButton>button {
        background-color: #1E8449;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #27AE60;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    st.title("Optimización de la plantación en zonas de reforestación mediante Algoritmo Genético")

    st.sidebar.header("Parámetros del algoritmo genético")

    polygon_id = st.sidebar.selectbox(
        "Selecciona un polígono",
        options=sorted(polygon_areas.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    pop_size = st.sidebar.slider("Tamaño de población (GA)", 10, 200, 60, 10)
    generations = st.sidebar.slider("Número de generaciones (GA)", 10, 200, 50, 10)
    pc = st.sidebar.slider("Probabilidad de cruce (pc)", 0.0, 1.0, 0.8, 0.05)
    pm = st.sidebar.slider("Probabilidad de mutación (pm)", 0.0, 0.2, 0.02, 0.01)
    use_existing = st.sidebar.checkbox("Usar plantas existentes del polígono", value=True)
    seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=10_000, value=123, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("Simulación Monte Carlo")
    n_sim = st.sidebar.slider("N simulaciones 1 ha", 100, 5000, 1000, 100)

    st.markdown(
        f"""
        ### Información del polígono seleccionado: {polygon_id}
        - Área: **{polygon_areas[polygon_id]:.2f} ha**
        """
    )

    objetivo, existentes, compras = compute_targets_and_compras(polygon_id)

    with st.expander("Ver objetivos, existentes y compras por especie"):
        import pandas as pd
        df = pd.DataFrame({
            "Especie": species,
            "Nombre": [species_names[s] for s in species],
            "Objetivo": [objetivo[s] for s in species],
            "Existentes": [existentes.get(s, 0) for s in species],
            "Compras": [compras[s] for s in species],
        })
        st.dataframe(df, use_container_width=True)

    ejecutar = st.button("Ejecutar AG + Simulación")

    if "results" not in st.session_state:
        st.session_state["results"] = None

    if ejecutar:
        with st.spinner("Ejecutando algoritmo genético y simulación Monte Carlo..."):
            best_ind, best_fit, history, (rows, cols), comp_sum, dev, sim_results = genetic_algorithm_polygon(
                polygon_id=polygon_id,
                pop_size=pop_size,
                generations=generations,
                pc=pc,
                pm=pm,
                use_existing=use_existing,
                seed=seed,
                with_simulation=True,
                n_sim=n_sim,
            )
        st.session_state["results"] = {
            "polygon_id": polygon_id,
            "best_ind": best_ind,
            "best_fit": best_fit,
            "history": history,
            "rows": rows,
            "cols": cols,
            "comp_sum": comp_sum,
            "dev": dev,
            "sim_results": sim_results,
            "n_sim": n_sim,
        }

    results = st.session_state.get("results", None)
    if results is not None:
        if results["polygon_id"] != polygon_id:
            st.info("Has cambiado de polígono. Vuelve a presionar **Ejecutar GA + Simulación** para este polígono.")
        else:
            sim_results = results["sim_results"]

            st.subheader(f"Resultados del GA para el polígono {polygon_id}")

            st.markdown(f"**Mejor fitness:** `{results['best_fit']:.4f}`")
            st.markdown(f"- Compatibilidad total (theta): `{results['comp_sum']:.4f}`")
            st.markdown(f"- Desviación de demanda total: `{results['dev']}`")

            fig_hist, ax_hist = plt.subplots()
            ax_hist.plot(range(1, len(results["history"]) + 1), results["history"], color="#1E8449")
            ax_hist.set_xlabel("Generación")
            ax_hist.set_ylabel("Mejor fitness")
            ax_hist.set_title("Historial de mejor fitness (GA)")
            ax_hist.grid(True)
            st.pyplot(fig_hist)

            from pandas import DataFrame
            counts = Counter(results["best_ind"])
            df_counts = DataFrame({
                "Especie": species,
                "Nombre": [species_names[s] for s in species],
                "Asignadas (GA)": [counts.get(s, 0) for s in species],
                "Objetivo": [objetivo[s] for s in species],
                "Diferencia": [counts.get(s, 0) - objetivo[s] for s in species],
            })
            st.markdown("#### Conteo de plantas por especie (mejor solución GA)")
            st.dataframe(df_counts, use_container_width=True)

            st.markdown("#### Mapa de asignación (códigos de especie)")
            fig_map = plot_assignment(
                results["best_ind"],
                results["rows"],
                results["cols"],
                title=f"Asignación GA - {polygon_id}"
            )
            st.pyplot(fig_map)

            st.markdown("---")
            st.subheader(f"Simulación Monte Carlo integrada (1 ha, {results['n_sim']} corridas)")

            st.markdown(
                f"- Valor esperado del **total de plantas** en 1 ha: `{sim_results['expected_total_plants']:.2f}`"
            )
            st.markdown(
                f"- Valor esperado de la **competencia total (θ)** en 1 ha: `{sim_results['expected_competition']:.4f}`"
            )

            df_sim = DataFrame({
                "Especie": species,
                "Nombre": [species_names[s] for s in species],
                "E[Existentes] 1 ha": [sim_results["expected_existing_per_species"][s] for s in species],
                "E[A suministrar] 1 ha": [sim_results["expected_purchases_per_species"][s] for s in species],
            })
            st.markdown("#### Valor esperado de plantas por especie (existentes y a suministrar en 1 ha)")
            st.dataframe(df_sim, use_container_width=True)

            st.markdown("#### Matriz de probabilidad de transición entre especies (P_ij)")
            P = sim_results["transition_prob"]
            mat = []
            for si in species:
                row = [P[si][sj] for sj in species]
                mat.append(row)
            df_P = DataFrame(mat, index=species, columns=species)
            st.dataframe(df_P, use_container_width=True)

            st.markdown("#### Orden sugerido de plantación (según efectos promedio de compatibilidad)")
            planting_order = sim_results["planting_order"]
            pretty_order = " → ".join([f"{s} ({species_names[s]})" for s in planting_order])
            st.markdown(pretty_order)


if __name__ == "__main__":
    main()