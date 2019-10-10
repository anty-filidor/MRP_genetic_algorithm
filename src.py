from datasets import Datasets

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


permutation = data.get_permutation(name)
print(permutation)
distance = data.distance_permutation(name, permutation)
print(distance)
print('aaa')
