import numpy as np
from pulp import *
import pandas as pd
import time


especies = ['AL', 'AS', 'ASC', 'AST', 'OC', 'OE', 'OR', 'OS', 'PL', 'YF']
nombres_especies = [
    'Agave lechuguilla', 'Agave salmiana', 'Agave scabra', 'Agave striata',
    'Opuntia cantabrigiensis', 'Opuntia engelmannii', 'Opuntia robusta',
    'Opuntia streptacantha', 'Prosopis laevigata', 'Yucca filifera'
]

# Densidades por hectárea
densidades_ha = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])

# Área del polígono P1
area_p1 = 1.28

# Plantas existentes en P1 aprox
plantas_existentes_p1 = np.array([8, 46, 16, 19, 11, 14, 18, 12, 15, 9])

# Demanda teórica total por especie en P1
demanda_teorica = (densidades_ha * area_p1).astype(int)

# Plantas NUEVAS necesarias = demanda - existentes
plantas_nuevas_demanda = demanda_teorica - plantas_existentes_p1
plantas_nuevas_demanda = np.maximum(plantas_nuevas_demanda, 0)

total_plantas_existentes = int(np.sum(plantas_existentes_p1))
total_plantas_nuevas = int(np.sum(plantas_nuevas_demanda))
total_nodos = total_plantas_existentes + total_plantas_nuevas

print("="*70)
print("MODELO MILP - P1 COMPLETO CON PLANTAS EXISTENTES")
print("="*70)
print(f"\nÁrea P1: {area_p1} ha")
print(f"Plantas teóricas totales: {int(np.sum(demanda_teorica))}")
print(f"Plantas EXISTENTES: {total_plantas_existentes}")
print(f"Plantas NUEVAS a comprar: {total_plantas_nuevas}")
print(f"Total nodos (posiciones): {total_nodos}")

print("\nDistribución por especie:")
df_ini = pd.DataFrame({
    "Especie": especies,
    "Existentes": plantas_existentes_p1,
    "Demanda_teorica": demanda_teorica,
    "Nuevas_necesarias": plantas_nuevas_demanda
})
print(df_ini.to_string(index=False))

# Matriz de compatibilidad
matriz_compatibilidad = np.array([
    [-0.05, -0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.05],
    [-0.05, -0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.05],
    [-0.05, -0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.05],
    [-0.05, -0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.05],
    [0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 0.25, 0.05],
    [0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 0.25, 0.05],
    [0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 0.25, 0.05],
    [0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 0.25, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.00]
])

# Modelo matematico
# CONJUNTOS

S = range(len(especies)) 
N_existentes = range(total_plantas_existentes)
N_nuevos = range(total_plantas_existentes, total_nodos)
N_todos = range(total_nodos)

print("\nConjuntos:")
print(f"  |S| = {len(S)} especies")
print(f"  |N_existentes| = {len(N_existentes)} nodos")
print(f"  |N_nuevos| = {len(N_nuevos)} nodos")
print(f"  |N_todos| = {len(N_todos)} nodos")

# VARIABLES DE DECISION

print("\nCreando variables...")

modelo = LpProblem("Asignacion_P1_Completo_Con_Existentes", LpMaximize)

# x[s,n] = 1 si especie s en nodo n
x = {}
for s in S:
    for n in N_todos:
        x[(s, n)] = LpVariable(f"x_{s}_{n}", cat="Binary")

# c[s,sp,n,m] solo para vecinos lineales (n y n+1)
c = {}
vecinos = []
for n in range(total_nodos - 1):
    m = n + 1
    vecinos.append((n, m))
    for s in S:
        for sp in S:
            c[(s, sp, n, m)] = LpVariable(
                f"c_{s}_{sp}_{n}_{m}", lowBound=0, upBound=1
            )

print(f"  Variables x: {len(x)}")
print(f"  Variables c: {len(c)}")
print(f"  Pares de vecinos (lineal): {len(vecinos)}")

#Respetar existentes

print("\nFijando nodos existentes (no se pueden mover)...")

asignacion_existentes = {}
indice_nodo = 0
for s in S:
    for _ in range(plantas_existentes_p1[s]):
        n = indice_nodo
        asignacion_existentes[n] = s
        for sp in S:
            if sp == s:
                modelo += x[(sp, n)] == 1
            else:
                modelo += x[(sp, n)] == 0
        indice_nodo += 1

print(f"  Nodos existentes fijados: {len(asignacion_existentes)}")

# FUNCIÓN OBJETIVO

print("\nDefiniendo función objetivo (compatibilidad total)...")

obj = LpAffineExpression()
for s in S:
    for sp in S:
        for n, m in vecinos:
            sigma = matriz_compatibilidad[sp, s]
            obj += sigma * c[(s, sp, n, m)]

modelo += obj

# RESTRICCIONES

print("Agregando restricciones...")

# R1: una especie por nodo
for n in N_todos:
    modelo += lpSum([x[(s, n)] for s in S]) == 1

# R2: demanda mínima SOLO en nodos nuevos (nuevas plantas)
for s in S:
    modelo += lpSum([x[(s, n)] for n in N_nuevos]) >= plantas_nuevas_demanda[s]

# R3: capacidad
modelo += lpSum([x[(s, n)] for s in S for n in N_todos]) <= total_nodos

# R4-6: linealización de compatibilidad
for s in S:
    for sp in S:
        for n, m in vecinos:
            modelo += c[(s, sp, n, m)] <= x[(s, n)]
            modelo += c[(s, sp, n, m)] <= x[(sp, m)]
            modelo += c[(s, sp, n, m)] >= x[(s, n)] + x[(sp, m)] - 1

print(f"  Total restricciones: {len(modelo.constraints)}")
print("\nResolviendo (CBC, puede tardar 10–30 min)...")
t0 = time.time()
estado = modelo.solve(PULP_CBC_CMD(msg=1, timeLimit=3600))
dt = time.time() - t0

print("\nEstado:", LpStatus[estado])
print("Tiempo (s):", round(dt, 2))

# RESULTADOS

if LpStatus[estado] in ["Optimal", "Integer Feasible"]:
    print("\nSOLUCIÓN ENCONTRADA.\n")

    asignacion = {}
    for n in N_todos:
        for s in S:
            if x[(s, n)].varValue == 1:
                asignacion[n] = s
                break

    conteo_total = np.zeros(len(S), dtype=int)
    conteo_nuevas = np.zeros(len(S), dtype=int)

    for n in N_todos:
        conteo_total[asignacion[n]] += 1
    for n in N_nuevos:
        conteo_nuevas[asignacion[n]] += 1

    print("Compatibilidad total:", value(modelo.objective))

    df_resumen = pd.DataFrame({
        "Especie": especies,
        "Existentes": plantas_existentes_p1,
        "Nuevas": conteo_nuevas,
        "Total": conteo_total,
        "Demanda_teorica": demanda_teorica,
        "Cumple?": [
            "✓" if conteo_total[i] >= demanda_teorica[i] else "✗"
            for i in range(len(S))
        ]
    })
    print("\nResumen final por especie:")
    print(df_resumen.to_string(index=False))
else:
    print("\nNo se encontró solución óptima (revisa tiempo/memoria).")