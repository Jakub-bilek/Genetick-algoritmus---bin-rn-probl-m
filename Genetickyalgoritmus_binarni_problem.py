import random
import numpy as np
import matplotlib.pyplot as plt

# 1. ÚČELOVÉ FUNKCE/FITNESS
def fitness_onemax(chromosome):
    """Pokud 0 tak nezmění pokud 1 tak ++"""
    return sum(chromosome)

def fitness_leading_ones(chromosome):
    """L->P 1++ 0break"""
    count = 0
    for bit in chromosome:
        if bit == 1:
            count += 1
        else:
            break
    return count

# 2. OPERÁTORY SELEKCE
def select_parent(population, fitnesses, selection_type='rank'):
    """Vybere rodiče podle typu buď ruleta(podle fitness má každý šanci) nebo rank(sorted podle fitness a šance podle pořadí)."""
    if selection_type == 'roulette':
        total_fit = sum(fitnesses)
        if total_fit == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fit)
        current = 0
        for ind, fit in zip(population, fitnesses):
            current += fit
            if current >= pick:
                return ind
        return population[-1]
        
    else:  # Rank selection (Pořadová selekce)
        sorted_pairs = sorted(zip(population, fitnesses), key=lambda x: x[1])
        n = len(sorted_pairs)
        total_rank_sum = n * (n + 1) / 2
        pick = random.uniform(0, total_rank_sum)
        
        current = 0
        for rank, (ind, _) in enumerate(sorted_pairs, start=1):
            current += rank
            if current >= pick:
                return ind
        return sorted_pairs[-1][0]

# 3. GENETICKÉ OPERÁTORY
def crossover_single_point(parent1, parent2):
    """Křížení rodičů náhodně vybere bod a vymění části chromozomů."""
    d = len(parent1)
    if d < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, d - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(chromosome, p_mut):
    """Bitová mutace procházející potomek bit po bitu s pravděpodobností p_mut."""
    return [1 - bit if random.random() < p_mut else bit for bit in chromosome]

# 4. GENETICKÝ ALGORITMUS
def run_genetic_algorithm(
    fitness_func, 
    dimension, 
    max_evaluations, 
    pop_size=30, 
    elitism_ratio=0.1, 
    p_mut=0.01, 
    selection_type='rank'
):
    """Spustí Genetický Algoritmus až do vyčerpání max_evaluations což vychází ze dánání."""
    population = [[random.randint(0, 1) for _ in range(dimension)] for _ in range(pop_size)]
    fitnesses = [fitness_func(ind) for ind in population]
    evaluations_count = pop_size

    history = []
    best_fitness = max(fitnesses)
    history.append(best_fitness)

    num_elites = max(1, int(pop_size * elitism_ratio))

    while evaluations_count < max_evaluations:
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1], reverse=True)

        new_population = []
        new_fitnesses = []

        # Elitismus
        for i in range(num_elites):
            new_population.append(pop_fit[i][0][:])
            new_fitnesses.append(pop_fit[i][1])

        # Křížení a mutace
        while len(new_population) < pop_size and evaluations_count < max_evaluations:
            p1 = select_parent(population, fitnesses, selection_type)
            p2 = select_parent(population, fitnesses, selection_type)

            off1, off2 = crossover_single_point(p1, p2)
            off1 = mutate(off1, p_mut)
            off2 = mutate(off2, p_mut)

            fit1 = fitness_func(off1)
            evaluations_count += 1
            new_population.append(off1)
            new_fitnesses.append(fit1)
            best_fitness = max(best_fitness, fit1)
            history.append(best_fitness)

            if len(new_population) < pop_size and evaluations_count < max_evaluations:
                fit2 = fitness_func(off2)
                evaluations_count += 1
                new_population.append(off2)
                new_fitnesses.append(fit2)
                best_fitness = max(best_fitness, fit2)
                history.append(best_fitness)

        population = new_population
        fitnesses = new_fitnesses

    while len(history) < max_evaluations:
        history.append(best_fitness)

    return history[:max_evaluations], best_fitness

# 5.BĚHY A VYHODNOCENÍ
def execute_experiments():
    dimensions = [10, 30, 100] 
    runs = 10             

    problems = [
        ('One-Max', fitness_onemax, 'blue'),
        ('Leading-Ones', fitness_leading_ones, 'orange')
    ]

    pop_size = 30
    elitism_ratio = 0.1  # 10 %
    p_mut = 0.01         # 1 %
    selection_method = 'rank'

    # Vytvoření okna: 2 řádky pro grafy + 1 spodní řádek pro tabulku
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, len(dimensions), height_ratios=[1, 1, 0.4])

    summary_stats = []

    # Procházíme oba problémy (vytvoří 2 řádky grafů)
    for p_idx, (prob_name, fit_func, color) in enumerate(problems):
        for d_idx, dim in enumerate(dimensions):
            max_evals = 100 * dim
            ax = fig.add_subplot(gs[p_idx, d_idx])

            print(f"---> Spouštím {prob_name} pro {dim}D (Limit: {max_evals} evaluací)")

            all_histories = []
            final_results = []

            for run in range(runs):
                random.seed(run * 42 + dim)
                history, best_fit = run_genetic_algorithm(
                    fitness_func=fit_func,
                    dimension=dim,
                    max_evaluations=max_evals,
                    pop_size=pop_size,
                    elitism_ratio=elitism_ratio,
                    p_mut=p_mut,
                    selection_type=selection_method
                )
                all_histories.append(history)
                final_results.append(best_fit)
                print(f"   [{prob_name}] Běh {run + 1}/{runs} - Najel: {best_fit}/{dim}")

            # Výpočet průměrné konvergence
            histories_array = np.array(all_histories)
            avg_history = np.mean(histories_array, axis=0)

            # Statistiky
            best_res = int(np.max(final_results))
            worst_res = int(np.min(final_results))
            mean_res = round(float(np.mean(final_results)), 2)
            median_res = round(float(np.median(final_results)), 2)
            std_res = round(float(np.std(final_results)), 2)

            summary_stats.append([
                f"{dim}D", prob_name, best_res, worst_res, mean_res, median_res, std_res
            ])

            # Vykreslení grafu
            ax.plot(avg_history, color=color, linewidth=2, label=f'{prob_name}')
            ax.axhline(y=dim, color='red', linestyle='--', label=f'Optimum ({dim})')
            ax.set_ylim(bottom=0, top=dim * 1.05)  # Začátek od 0
            ax.set_title(f"{prob_name} - {dim}D")
            ax.set_xlabel("Počet evaluací")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            ax.legend()

    # Tabulka dole
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    headers = ["Dimenze", "Problém", "Nejlepší", "Nejhorší", "Průměr", "Medián", "Směrodatná odch."]
    
    #  Tabulka v okně
    table = ax_table.table(
        cellText=summary_stats,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    execute_experiments()