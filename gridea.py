# encoding: utf-8
# The main definition of the evolutionary algorithm
import argparse
import initialize
import json
import logging
import multiprocessing
import network
import numba
import numpy as np
import random
import scoring
import time
import utils
from twisted.internet import reactor

__author__ = 'Jason Ansel'

log = logging.getLogger(__name__)


def mutate(solution):
    """
    Unoptimized mutation operator included for clarity. Optimized version is
    below.  Randomly shifts one element forward and another backward.

    :param solution: input/output permutation of the points in the puzzle
    """
    shift_fwd_idx = random.randint(0, solution.shape[0])
    shift_back_idx = random.randint(0, solution.shape[0])
    utils.shift_to(shift_fwd_idx, 0, solution)
    utils.shift_to(shift_back_idx, solution.shape[0] - 1, solution)


def crossover(solution_a, solution_b, solution_out, puzzle):
    """
    Unoptimized mutation operator included for clarity.  Optimized version
    below.  Randomly draws a line across the puzzle board, takes the
    points above the line from solution_a and the points below the line
    from solution_b.

    :param solution_a: input, permutation of the points in the puzzle
    :param solution_b: input, permutation of the points in the puzzle
    :param solution_out: output, permutation of the points in the puzzle
    :param puzzle: the puzzle board
    :return:
    """
    split = random.random()
    p = random.random()
    i_mult = p / float(puzzle.shape[0])
    j_mult = (1.0 - p) / float(puzzle.shape[1])
    out_idx = 0
    for point in solution_a:
        i, j = utils.split_point(point)
        if i_mult * i + j_mult * j <= split:
            solution_out[out_idx] = point
            out_idx += 1
    for point in solution_b:
        i, j = utils.split_point(point)
        if i_mult * i + j_mult * j > split:
            solution_out[out_idx] = point
            out_idx += 1


@numba.autojit(nopython=True, nogil=True)
def copy_and_mutate(solution_in, solution_out, random_state):
    """
    An optimized combination of utils.copy_1d() and
    mutate().  Makes only a single pass over the list.

    :param solution_in: array representing population member to read
    :param solution_out: array representing population member to write
    :param random_state: for utils.xorshift(), a faster randint
    """
    size = solution_out.shape[0]
    shift_fwd_idx = utils.xorshift(random_state) % (size - 1) + 1
    shift_back_idx = utils.xorshift(random_state) % (size - 1) + 1
    out_idx = 1
    for point in solution_in:
        if out_idx == shift_fwd_idx:
            solution_out[0] = point
            shift_fwd_idx = 0
        elif out_idx == shift_back_idx:
            solution_out[size - 1] = point
            shift_back_idx = 0
        else:
            solution_out[out_idx] = point
            out_idx += 1


@numba.autojit(nopython=True, nogil=True)
def crossover_and_mutate(solution_a, solution_b, solution_out, random_state,
                         puzzle):
    """
    An optimized combination of crossover() and mutate() that does only
    a single pass over the list.  The floating point math in crossover()
    has been replaced with integer math for performance.

    :param solution_a: input, permutation of the points in the puzzle
    :param solution_b: input, permutation of the points in the puzzle
    :param solution_out: output, permutation of the points in the puzzle
    :param random_state: for utils.xorshift(), a faster randint
    :param puzzle: the puzzle board, used only for dimensions
    """
    large_number = 10000  # to approximate 1.0 in floating point math
    split = utils.xorshift(random_state) % large_number
    p = utils.xorshift(random_state) % large_number
    i_mult = p // puzzle.shape[0]
    j_mult = (large_number - p) // puzzle.shape[1]
    size = solution_out.shape[0]
    shift_fwd_idx = utils.xorshift(random_state) % (size - 1) + 1
    shift_back_idx = utils.xorshift(random_state) % (size - 1) + 1

    out_idx = 1
    for point in solution_a:
        i, j = utils.split_point(point)
        if i_mult * i + j_mult * j <= split:
            if out_idx == shift_fwd_idx:
                solution_out[0] = point
                shift_fwd_idx = 0
            elif out_idx == shift_back_idx:
                solution_out[size - 1] = point
                shift_back_idx = 0
            else:
                solution_out[out_idx] = point
                out_idx += 1
    for point in solution_b:
        i, j = utils.split_point(point)
        if i_mult * i + j_mult * j > split:
            if out_idx == shift_fwd_idx:
                solution_out[0] = point
                shift_fwd_idx = 0
            elif out_idx == shift_back_idx:
                solution_out[size - 1] = point
                shift_back_idx = 0
            else:
                solution_out[out_idx] = point
                out_idx += 1


@numba.jit('void(uint32[:,:], uint32, uint8[:,:], uint32[:])',
           nopython=True, nogil=True)
def spawn_new_population_members(population, spawn_count, puzzle,
                                 random_state):
    """
    Create new population members using 50% crossover_and_mutate() and
    50% copy_and_mutate().  New population members are written to the last
    `spawn_count` rows of `population`.

    :param population: 2D matrix, rows are members with score as first element
    :param spawn_count: count of population members to generate
    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :param random_state: for utils.xorshift(), a faster randint
    """
    split_point = population.shape[0] - spawn_count
    for k in range(split_point, population.shape[0]):
        a = utils.xorshift(random_state) % split_point
        if k % 2 == 0:
            b = utils.xorshift(random_state) % split_point
            crossover_and_mutate(population[a], population[b], population[k],
                                 random_state, puzzle)
        else:
            copy_and_mutate(population[a], population[k], random_state)


class GrideaWorker(object):
    def __init__(self, args):
        """
        Constructor

        :param args: namespace generated by argparse
        """
        self.args = args
        self.network = network.GrideaProtocol()
        self.random_state = np.array([random.getrandbits(32)
                                      for _ in range(4)], dtype=np.uint32)
        self.puzzle_id = None
        self.puzzle = None
        self.puzzle_scratch = None
        self.puzzle_sum = 0
        self.population = None

    def generation(self):
        """
        Run one generation of our evolutionary algorithm
        """

        # Partially sort population such that all members before the index
        # `pop_size` score better than all members after `pop_size`
        utils.divide_population(self.population, self.args.pop_size)

        # Replace all population members after index `pop_size` with newly
        # spawned solutions
        spawn_new_population_members(self.population[:, 1:],
                                     self.args.spawn_count, self.puzzle,
                                     self.random_state)

        # Fill in scores (as the first element of the row) for all population
        # members after index `pop_size`
        scoring.score_population(self.puzzle, self.puzzle_sum,
                                 self.puzzle_scratch,
                                 self.population[self.args.pop_size:])

    def solve(self, puzzle_metadata):
        """
        Main entry point to this evolutionary algorithm to solve a given puzzle
        instance.

        :param puzzle_metadata: puzzle instance in challenge API format
        :return: a solved puzzle in challenge API format
        """
        stop_time = time.time() + self.args.limit
        next_callback = time.time() + self.args.share_freq

        self.puzzle_id = puzzle_metadata['id']
        self.puzzle = np.array(puzzle_metadata['puzzle'], dtype=np.uint8)
        self.puzzle_scratch = np.ndarray(self.puzzle.shape, dtype=np.uint8)
        self.puzzle_sum = self.puzzle.sum()
        self.population = initialize.init_population(
            self.puzzle, self.puzzle_sum, self.puzzle_scratch,
            self.args.pop_size + self.args.spawn_count)

        while time.time() < stop_time:
            self.generation()

            if time.time() > next_callback:
                # Share best result with network
                best_idx = self.population[:, 0].argmin()
                self.network.best(self.puzzle_id,
                                  int(self.population[best_idx, 0]),
                                  map(int, self.population[best_idx, 1:]))
                next_callback = time.time() + self.args.share_freq

        best_idx = self.population[:, 0].argmin()
        self.network.best(self.puzzle_id, int(self.population[best_idx, 0]),
                          map(int, self.population[best_idx, 1:]))
        return scoring.expand_solution(self.puzzle,
                                       self.population[best_idx, 1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default=9.6, type=float,
                        help='number of seconds to run for')
    parser.add_argument('--share-freq', default=0.5, type=float, metavar='SF',
                        help='how often in seconds to report best')
    parser.add_argument('--pop-size', default=1000, type=int, metavar='N',
                        help='number of solutions kept in the population')
    parser.add_argument('--spawn-count', default=100, type=int, metavar='K',
                        help='number of new solutions added each generation')
    parser.add_argument('--link',
                        help='join this network to some other hostname:port')
    parser.add_argument('--port', '-p', type=int, default=8099,
                        help='port to listen on')
    parser.add_argument('--debug', '-v', action='store_true',
                        help='print verbose debugging output')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--filename',
                       help='solve a puzzle from a local JSON filename')
    group.add_argument('--workers', type=int,
                       help='create a network with a given number of workers')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    if args.filename:
        puzzle = json.load(open(args.filename))
        network.GlobalBest.reset(puzzle['id'])
        result = GrideaWorker(args).solve(puzzle)
        json.dump(result, open(args.filename + '.result', 'w'))
        print 'score={}, result written to {}.result'.format(len(result),
                                                             args.filename)
    else:
        pool = multiprocessing.Pool(args.workers)
        jobs = pool.imap(run_child_process, [args] * args.workers)
        network.listen(args.port)
        if args.link:
            network.connect(args.link)
        reactor.run()
        for _ in jobs:
            pass  # Empty job queue
        pool.terminate()
        pool.join()


def run_child_process(args):
    network.connect('localhost:{}'.format(args.port), GrideaWorker(args))
    reactor.run()


if __name__ == '__main__':
    main()
