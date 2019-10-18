## Genethic algorithm for travelling salesman problem

Program includes two classes:
* datasets
* genethic algorithm

Datasets used in experiments are avaliable [here](https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/STSP.html).

Usage of algorithm:

`GA = GeneticAlgorithm(data, name)`

`stats = GA(population_size=50, size_of_elite=5, mutation_ratio=0.01, epochs=50)`

Kind of selection: roulette with elitism 

Kind of crossing over: ordered

Kind of mutation: swap
