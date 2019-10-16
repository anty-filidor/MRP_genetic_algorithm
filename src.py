from datasets import Datasets
import operator
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

path = '/Users/michal/PycharmProjects/MRP/datasets/*.tsp'
data = Datasets(path)

# name = 'ali535'
name = 'berlin11_modified'
# name = 'berlin52'
# name = 'fl417'
# name = 'gr666'
# name = 'kroA100'
# name = 'kroA150'
# name = 'nrw1379'
# name = 'pr2392'

# print(data.datasets[name].loc[4]['x'])
# print(data._distance_two_cities(name, 1, 2))
# print(data.distance_permutation(name, [1, 2, 3, 4]))

# permutation = data.get_permutation(name)
# print(permutation)
# distance = data.distance_permutation(name, permutation)
# print(distance)

from genetic_algorithm import GeneticAlgorithm

GA = GeneticAlgorithm(data, name)
GA(population_size=20, elite_size=1, mutation_ratio=0.01, epochs=100)


def generate_population(population_length):
    population = []
    for i in range(0, population_length):
        population.append(data.get_permutation(name))
    return population


def rank_population(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = data.distance_permutation(name, population[i])
    ranked_population = np.array(sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False))
    # return dataframe with rows sorted by shortest distance and index of permutation
    return pd.DataFrame(ranked_population, columns=['index', 'fitness'])


def select_individuals(ranked_population, size_of_elite):
    selection_results = []
    df = ranked_population.copy()
    df['cum_sum'] = df.fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

    for i in range(0, size_of_elite):
        selection_results.append(int(ranked_population.iloc[i]['index']))
    for i in range(0, len(ranked_population) - size_of_elite):
        pick = 100 * random.random()
        for i in range(0, len(ranked_population)):
            if pick <= df.iat[i, 3]:
                selection_results.append(int(ranked_population.iloc[i]['index']))
                break
    return selection_results


def crossing_over(parent1, parent2):
    cutting_place_a = int(random.random() * len(parent1))
    cutting_place_b = int(random.random() * len(parent1))

    start_place = min(cutting_place_a, cutting_place_b)
    stop_place = max(cutting_place_a, cutting_place_b)

    genome_a = []
    for i in range(start_place, stop_place):
        genome_a.append(parent1[i])

    gene_b = [item for item in parent2 if item not in genome_a]

    child = genome_a + gene_b
    return child


def mutation(individual, ratio):
    for gene_index in range(len(individual)):
        if random.random() < ratio:
            swap_with = int(random.random() * len(individual))
            gene1 = individual[gene_index]

            individual[gene_index] = individual[swap_with]
            individual[swap_with] = gene1
    return individual


def breed_population(population, selection_results, size_of_elite):
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])

    children = []

    length = len(matingpool) - size_of_elite
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, size_of_elite):
        children.append(matingpool[i])

    for i in range(0, length):
        child = crossing_over(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate_population(population, ratio):
    mutated_population = []
    for individual in range(0, len(population)):
        mutated_population.append(mutation(population[individual], ratio))
    return mutated_population


def create_next_generation(currentGen, eliteSize, mutationRate):
    popRanked = rank_population(currentGen)
    selected_routes = select_individuals(popRanked, eliteSize)
    children = breed_population(currentGen, selected_routes, eliteSize)
    nextGeneration = mutate_population(children, mutationRate)
    return nextGeneration


def genetic_algorithm(population, population_size, elite_size, mutation_ratio, epochs):
    #  Initialise lists to keep best results in each epoch
    best_routes = []

    #  Initialise populations
    population = generate_population(population_size)
    best_routes.append(rank_population(population).iloc[0].fitness)
    print("Initial distance: " + str(best_routes[0]))

    for i in range(0, epochs):
        population = create_next_generation(population, elite_size, mutation_ratio)
        best_routes.append(rank_population(population).iloc[0].fitness)

    print("Final distance: " + str(best_routes[epochs]))

    plt.plot(best_routes)
    plt.ylabel('Distance')
    plt.xlabel('Epoch')
    plt.show()

    return population[int(rank_population(population).iloc[0]['index'])]

'''
genetic_algorithm(1, population_size=20, elite_size=1, mutation_ratio=0.01, epochs=20)



population = generate_population(15)
print(population)

ranked_routes = rank_population(population)
print(ranked_routes, 'aa', rank_population(population).iloc[0].fitness)

selected_routes = select_individuals(ranked_routes, 1)
print(selected_routes)

children = breed_population(population, selected_routes, 1)
print(children)

mutated_children = mutate_population(children, 0.01)
print(mutated_children)

'''