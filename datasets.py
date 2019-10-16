import numpy as np
from glob import glob
import pandas as pd
import random


class Datasets:
    def __init__(self, path_to_datasets):
        self.datasets = {}  # dictionary to keep datasets
        paths = glob(path_to_datasets)

        for path in paths:
            # save dataset name
            dataset_name = path.split('/')[-1].split('.')[0]

            # initialise empty containers
            metadata = []
            num_of_rows_to_skip = 1

            # count num of rows to skip and save metadata
            with open(path, 'r') as file:
                # omit trash and save it in metadata
                line = file.readline()
                while line and 'NODE_COORD_SECTION' not in line:
                    line = file.readline()
                    metadata.append(line)
                    num_of_rows_to_skip += 1

            # read content to pandas dataframe, drop last row because of 'EOF', set indexing
            dataset = pd.read_csv(path, delim_whitespace=True, skiprows=num_of_rows_to_skip, names=['id', 'x', 'y'])
            dataset = dataset.drop(dataset.index[-1])
            dataset = dataset.drop(columns='id')
            dataset.index += 1

            # add metadata to dataframe
            dataset.metadata = metadata

            # append metadata to dictionary
            self.datasets.update({dataset_name: dataset})

    def _distance_two_cities(self, dataset, city1, city2):
        return np.sqrt(abs(self.datasets[dataset].loc[city1]['x']-self.datasets[dataset].loc[city2]['x'])**2 +
                       abs(self.datasets[dataset].loc[city1]['y']-self.datasets[dataset].loc[city2]['y'])**2)

    def distance_permutation(self, name, permutation):
        distance = 0
        for city1, city2 in zip(permutation[:-1], permutation[1:]):
            distance += self._distance_two_cities(name, city1, city2)
        return distance

    def get_permutation(self, name):
        permutation = random.sample(range(1, self.datasets[name].shape[0]+1), self.datasets[name].shape[0])
        permutation.append(permutation[0])
        return permutation


# Example of usage

# path = '/Users/michal/PycharmProjects/MRP/datasets/*.tsp'
# data = Datasets(path)
# name = 'berlin11_modified'

# print(data.datasets[name].loc[4]['x'])
# print(data._distance_two_cities(name, 1, 2))
# print(data.distance_permutation(name, [1, 2, 3, 4]))

# permutation = data.get_permutation(name)
# print(permutation)
# distance = data.distance_permutation(name, permutation)
# print(distance)
