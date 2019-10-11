from datasets import Datasets
import operator

path = '/Users/michal/PycharmProjects/MRP/datasets/*.tsp'
data = Datasets(path)

name = 'ali535'
# name = 'berlin11_modified'
# name = 'berlin52'
# name = 'fl417'
# name = 'gr666'
# name = 'kroA100'
# name = 'kroA150'
# name = 'nrw1379'
# name = 'pr2392'

# print(data.datasets[name].loc[4]['x'])
# print(data.distance_two_cities(name, 1, 2))
# print(data.distance_permutation(name, [1, 2, 3, 4]))

# permutation = data.get_permutation(name)
# print(permutation)
# distance = data.distance_permutation(name, permutation)
# print(distance)


def generate_population(population_length):
    population = []
    for i in range(0, population_length):
        population.append(data.get_permutation(name))
    return population


def rank_routes(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = data.distance_permutation(name, population[i])
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False)


population = generate_population(10)
print(population)

ranked_routes = rank_routes(population)
print(ranked_routes)
