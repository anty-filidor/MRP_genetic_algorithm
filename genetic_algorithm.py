from datasets import Datasets
import operator
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt


class GeneticAlgorithm:
    def __init__(self, datasets, name):
        self.datasets = datasets
        self.name = name

    def generate_population(self, population_length):
        population = []
        for i in range(0, population_length):
            population.append(self.datasets.get_permutation(self.name))
        return population

    def rank_population(self, population):
        fitness_results = {}
        for i in range(0, len(population)):
            fitness_results[i] = self.datasets.distance_permutation(self.name, population[i])
        ranked_population = np.array(sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False))
        # return dataframe with rows sorted by shortest distance and index of permutation
        return pd.DataFrame(ranked_population, columns=['index', 'fitness'])

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def mutation(individual, ratio):
        for gene_index in range(len(individual)):
            if random.random() < ratio:
                swap_with = int(random.random() * len(individual))
                gene1 = individual[gene_index]

                individual[gene_index] = individual[swap_with]
                individual[swap_with] = gene1
        return individual

    def breed_population(self, population, selection_results, size_of_elite):
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
            child = self.crossing_over(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def mutate_population(self, population, ratio):
        mutated_population = []
        for individual in range(0, len(population)):
            mutated_population.append(self.mutation(population[individual], ratio))
        return mutated_population

    def __call__(self, population_size, elite_size, mutation_ratio, epochs):
        #  Initialise lists to keep best results in each epoch
        best_routes = []

        #  Initialise populations
        population = self.generate_population(population_size)
        best_routes.append(self.rank_population(population).iloc[0].fitness)
        print("Initial distance: " + str(best_routes[0]))

        for i in range(0, epochs):
            popRanked = self.rank_population(population)
            selected_routes = self.select_individuals(popRanked, elite_size)
            children = self.breed_population(population, selected_routes, elite_size)
            population = self.mutate_population(children, mutation_ratio)

            best_routes.append(self.rank_population(population).iloc[0].fitness)

        print("Final distance: " + str(best_routes[epochs]))

        plt.plot(best_routes)
        plt.ylabel('Distance')
        plt.xlabel('Epoch')
        plt.show()

        return population[int(self.rank_population(population).iloc[0]['index'])]

