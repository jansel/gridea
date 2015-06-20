# encoding: utf-8
# Expand the permutation representation to full list of boxes or count (score)
# the number of boxes that would be required by a given permutation
import numba
import numpy as np
import utils

__author__ = 'Jason Ansel'


def expand_solution(puzzle, solution):
    """
    Our solution representation is a permutation of points which determine the
    order in which to draw squares on the solution matrix using each point as
    the top left corner of a square which is drawn to fill all available space.

    expand_solution() draws squares in the order specified by `solution`, and
    returns the list of resulting squares drawn in challenge API format.

    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :param solution: a permutation of points in packed uint32 form
    :return: list of [{'X': x, 'Y': y, 'Size': n}, ...] boxes
    """
    height, width = puzzle.shape
    scratch = np.copy(puzzle)
    boxes = []
    # in scratch[x, y]:
    #    0 means tile is not part of the puzzle shape
    #    1 means tile is empty
    #    2 means tile is top left corner of box
    #    3 means tile is filled by box
    for point in solution:
        i, j = utils.split_point(point)
        assert scratch[i, j] > 0
        if scratch[i, j] == 1:
            n = 1
            hit_wall = False
            # Keep expanding the box until we hit a wall
            while not hit_wall and i + n < height and j + n < width:
                for k in range(n + 1):
                    if scratch[i + n, j + k] != 1:
                        hit_wall = True
                        break
                    if scratch[i + k, j + n] != 1:
                        hit_wall = True
                        break
                if not hit_wall:
                    for k in range(n + 1):
                        scratch[i + n, j + k] = 3
                        scratch[i + k, j + n] = 3
                    n += 1
            if n > 1:  # Draw 1x1 boxes later, as they wont help the score
                scratch[i, j] = 2
                boxes.append((i, j, n))

    # Draw all 1x1 boxes in a final pass
    for i in range(height):
        for j in range(width):
            if scratch[i, j] == 1:
                boxes.append((i, j, 1))

    return [{'X': x, 'Y': y, 'Size': size} for y, x, size in boxes]


@numba.autojit(nopython=True, nogil=True)
def score_solution(puzzle, puzzle_sum, scratch, solution):
    """
    Optimized version of expand_solution (10x to 100x faster) that only counts
    the number of squares needed without drawing them.

    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :param puzzle_sum: count of number of tiles in the puzzle
    :param scratch: scratchpad memory (same shape as puzzle)
    :param solution: a permutation of points in packed uint32 form
    :return: integer count of squared used by solution
    """
    height, width = puzzle.shape
    utils.copy_2d(puzzle, scratch)
    # in scratch[i, j]:
    #    0 means tile is not part of the puzzle shape
    #    1 means tile is empty
    #    2 means tile is top left corner of box
    #    3 means tile is filled by box
    score = 0
    tiles_used = 0
    for point in solution:
        i, j = utils.split_point(point)
        # Unrolled version of the loop in expand_solution (~30% speedup)
        if (scratch[i, j] == 1 and scratch[i, j + 1] == 1 and
                scratch[i + 1, j] == 1 and scratch[i + 1, j + 1] == 1):
            scratch[i + 0, j + 0] = 2
            scratch[i + 0, j + 1] = 3
            scratch[i + 1, j + 0] = 3
            scratch[i + 1, j + 1] = 3
            n = 2
            if (i + 2 < height and j + 2 < width and
                    scratch[i + 0, j + 2] == 1 and
                    scratch[i + 1, j + 2] == 1 and
                    scratch[i + 2, j + 0] == 1 and
                    scratch[i + 2, j + 1] == 1 and
                    scratch[i + 2, j + 2] == 1):
                scratch[i + 0, j + 2] = 3
                scratch[i + 1, j + 2] = 3
                scratch[i + 2, j + 0] = 3
                scratch[i + 2, j + 1] = 3
                scratch[i + 2, j + 2] = 3
                n = 3
                if (i + 3 < height and j + 3 < width and
                        scratch[i + 3, j + 0] == 1 and
                        scratch[i + 3, j + 1] == 1 and
                        scratch[i + 3, j + 2] == 1 and
                        scratch[i + 3, j + 3] == 1 and
                        scratch[i + 0, j + 3] == 1 and
                        scratch[i + 1, j + 3] == 1 and
                        scratch[i + 2, j + 3] == 1):
                    scratch[i + 0, j + 3] = 3
                    scratch[i + 1, j + 3] = 3
                    scratch[i + 2, j + 3] = 3
                    scratch[i + 3, j + 0] = 3
                    scratch[i + 3, j + 1] = 3
                    scratch[i + 3, j + 2] = 3
                    scratch[i + 3, j + 3] = 3
                    n = 4
                    hit_wall = False
                    while not hit_wall and i + n < height and j + n < width:
                        for k in range(n + 1):
                            if scratch[i + n, j + k] != 1:
                                hit_wall = True
                                break
                            if scratch[i + k, j + n] != 1:
                                hit_wall = True
                                break
                        if not hit_wall:
                            for k in range(n + 1):
                                scratch[i + n, j + k] = 3
                                scratch[i + k, j + n] = 3
                            n += 1
            score += 1
            tiles_used += n * n

    # Cost of 1x1 boxes, we can calculate the number needed without iterating
    score += puzzle_sum - tiles_used
    return score


@numba.jit('void(uint8[:, :], uint32, uint8[:, :], uint32[:,:])',
           nopython=True, nogil=True)
def score_population(puzzle, puzzle_sum, puzzle_scratch, population):
    """
    Score each member of the population, writing scores to the first element.

    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :param puzzle_sum: count of number of tiles in the puzzle
    :param puzzle_scratch: scratchpad memory (same shape as puzzle)
    :param population: 2D matrix where rows are population members, score is
                       stored in the first element and the perumation of points
                       in the rest
    """
    for k in range(population.shape[0]):
        population[k, 0] = score_solution(puzzle, puzzle_sum, puzzle_scratch,
                                          population[k, 1:])


def get_valid_points(puzzle):
    """
    Return a list of points (in compacted uint32 form) in the puzzle that:
        1) Are inside the bounds of the puzzle
        2) Can expand down and to the right to form at least a 2x2 square

    Note that only points with a maximum possible square size > 1 are
    interesting and included. The scoring function will automatically fill
    in 1x1 squares over empty tiles.
    """
    valid_points = []
    for i in range(puzzle.shape[0] - 1):
        for j in range(puzzle.shape[1] - 1):
            two_box = (puzzle[i + 0, j + 0] + puzzle[i + 0, j + 1] +
                       puzzle[i + 1, j + 0] + puzzle[i + 1, j + 1])
            if two_box == 4:
                valid_points.append((i, j))
    valid_points_compact = np.array([(i << 16) + j for i, j in valid_points],
                                    dtype=np.uint32)
    assert list(map(utils.split_point, valid_points_compact)) == valid_points
    assert len(set(valid_points_compact)) == len(valid_points)
    return valid_points_compact


def calculate_max_sizes(puzzle):
    """
    Calculate the maximum sized squares that could be drawn from each
    valid point.  Used by heuristics in forming the initial population.

    :param puzzle: a 2D matrix of bools representing the puzzle (0 is a wall)
    :return: A list of (N, point) where N is the largest square drawable
             from point
    """
    height, width = puzzle.shape
    scratch = np.copy(puzzle)
    boxes = []
    for i in range(height):
        for j in range(width):
            if puzzle[i, j] == 1:
                # Keep expanding the box until we hit a wall
                n = 1
                hit_wall = False
                while not hit_wall and i + n < height and j + n < width:
                    for k in range(n + 1):
                        if scratch[i + n, j + k] != 1:
                            hit_wall = True
                            break
                        if scratch[i + k, j + n] != 1:
                            hit_wall = True
                            break
                    if not hit_wall:
                        n += 1
                if n > 1:
                    boxes.append((n, (i << 16) + j))
    return boxes
