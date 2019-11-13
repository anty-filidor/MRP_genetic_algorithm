import operator
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(self, datasets, name):
        self.datasets = datasets
        self.name = name

    def _generate_population(self, population_length):
        population = []
        for i in range(0, population_length):
            population.append(self.datasets.get_permutation(self.name))
        return population

    def _rank_population(self, population):
        fitness_results = {}
        for i in range(0, len(population)):
            fitness_results[i] = self.datasets.distance_permutation(self.name, population[i])
        ranked_population = np.array(sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False))
        # return dataframe with rows sorted by shortest distance and index of permutation
        return pd.DataFrame(ranked_population, columns=['index', 'fitness'])

    @staticmethod
    def _select_individuals(ranked_population, size_of_elite):
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
    def _crossing_over(parent1, parent2):
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
    def _mutation(individual, ratio):
        for gene_index in range(len(individual)):
            if random.random() < ratio:
                swap_with = int(random.random() * len(individual))
                gene1 = individual[gene_index]

                individual[gene_index] = individual[swap_with]
                individual[swap_with] = gene1
        return individual

    def _breed_population(self, population, selection_results, size_of_elite):
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
            child = self._crossing_over(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def _mutate_population(self, population, ratio, size_of_elite):
        mutated_population = []
        for individual in range(0, size_of_elite):
            mutated_population.append(population[individual])
        for individual in range(size_of_elite, len(population)):
            mutated_population.append(self._mutation(population[individual], ratio))
        return mutated_population

    def __call__(self, population_size, size_of_elite, mutation_ratio, epochs, plot_figure=True):
        #  Initialise lists to keep best results in each epoch
        best_distance = []
        best_route = []

        #  Initialise population
        population = self._generate_population(population_size)

        #  Update stats lists
        best_for_now = self._rank_population(population).iloc[0]
        best_distance.append(best_for_now.fitness)
        best_route.append(population[int(best_for_now['index'])])

        #  Main loop
        for i in tqdm(range(0, epochs)):
            ranked_population = self._rank_population(population)
            individuals_to_couple = self._select_individuals(ranked_population, size_of_elite)
            children = self._breed_population(population, individuals_to_couple, size_of_elite)
            population = self._mutate_population(children, mutation_ratio, size_of_elite)

            #  Update stats lists
            best_for_now = self._rank_population(population).iloc[0]
            best_distance.append(best_for_now.fitness)
            best_route.append(population[int(best_for_now['index'])])

        if plot_figure:
            plt.plot(best_distance)
            plt.ylabel('Distance')
            plt.xlabel('Epoch')
            plt.show()

        return dict(zip(best_distance, best_route))

