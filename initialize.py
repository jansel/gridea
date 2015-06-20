# encoding: utf-8
# Heuristics used to form the initial population of the evolutionary algorithm
import numba
import numpy as np
import scoring
import utils

__author__ = 'Jason Ansel'


def init_population(puzzle, puzzle_sum, puzzle_scratch, count):
    """
    Create an initial population for our evolutionary algorithm.  Uses
    different heuristics based on X, Y, and N, where N is the maximum sized
    box that could be drawn from a point.

    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :param puzzle_sum: count of number of tiles in the puzzle
    :param puzzle_scratch: scratchpad memory (same shape as puzzle)
    :param count: size of the population to generate
    :return: population 2D matrix, rows are members with score as first element
    """
    max_sizes = scoring.calculate_max_sizes(puzzle)
    population = np.zeros((count, len(max_sizes) + 1),
                          dtype=np.uint32)

    # Seed population with heuristics that sort points different ways
    heuristics = make_heuristic_list()
    for k, heuristic in enumerate(heuristics):
        max_sizes.sort(key=heuristic)
        population[k, 1:] = [point for n, point in max_sizes]
    scoring.score_population(puzzle, puzzle_sum, puzzle_scratch,
                             population[:len(heuristics)])
    fill_population_repeating(population, len(heuristics))
    # log.debug('best of %d heuristics is %d', n_heuristics,
    #           population[:, 0].min())
    return population


def make_angle_heuristic(pi, pj, pn):
    """
    Create a simple heuristic that uses different weights of i, j, and n.

    :param pi: weight for i
    :param pj: weight for j
    :param pn: weight for n (largest square possible at i, j)
    :return: function usable as a sort key
    """
    def key(n_and_point):
        (n, point) = n_and_point
        i, j = utils.split_point(point)
        return i * pi + j * pj - n * pn
    return key


def make_heuristic_list():
    """
    Build a list of heuristics that will be used to sort solutions different
    ways.

    :return: list of functions usable as a sort key
    """
    heuristics = [lambda n_p: (-n_p[0], (n_p[1] >> 16), (n_p[1] & 0xffff)),
                  lambda n_p: (-n_p[0], (n_p[1] & 0xffff), (n_p[1] >> 16)),
                  lambda n_p: ((n_p[1] >> 16), -n_p[0], (n_p[1] & 0xffff)),
                  lambda n_p: ((n_p[1] & 0xffff), -n_p[0], (n_p[1] >> 16)),
                  lambda n_p: ((n_p[1] >> 16), (n_p[1] & 0xffff)),
                  lambda n_p: ((n_p[1] & 0xffff), (n_p[1] >> 16))]
    samples = 50
    for ij in range(samples):
        for ratio in range(5):
            split = ij / float(samples - 1)
            heuristics.append(make_angle_heuristic(split, 1.0 - split, ratio))
    return heuristics


@numba.autojit(nopython=True, nogil=True)
def fill_population_repeating(population, num_heuristics):
    """
    Fill population[num_heuristics:] by duplicating population[:num_heuristics]
    repeatedly

    :param population: 2D matrix, rows are members with score as first element
    :param num_heuristics: index to start filling from
    """
    for k in range(num_heuristics, population.shape[0]):
        utils.copy_1d(population[k % num_heuristics], population[k])
