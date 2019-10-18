from datasets import Datasets
from genetic_algorithm import GeneticAlgorithm

path = '/Users/michal/PycharmProjects/MRP/datasets/*.tsp'
data = Datasets(path)

# name = 'ali535'
# name = 'berlin11_modified'
name = 'berlin52'
# name = 'fl417'
# name = 'gr666'
# name = 'kroA100'
# name = 'kroA150'
# name = 'nrw1379'
# name = 'pr2392'

GA = GeneticAlgorithm(data, name)
stats = GA(population_size=50, size_of_elite=5, mutation_ratio=0.01, epochs=50)

for iterator, log in enumerate(stats.items()):
    print('EPOCH {}'.format(iterator))
    print('\t\tbest distance - {}'.format(log[0]))
    print('\t\tbest route - {}'.format(log[1]))
    print('\n')
