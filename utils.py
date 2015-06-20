# encoding: utf-8
# Some general helper functions
import numba

__author__ = 'Jason Ansel'


@numba.jit('uint32(uint32[:])', nopython=True, nogil=True)
def xorshift(random_state):
    """
    Faster random number generator that we can map to numba.jit code.
    See: http://wki.pe/Xorshift

    :param random_state: numpy.ndarray(4, dtype=numpy.uint32)
    :return: a random uint32 number
    """
    t = random_state[0] ^ (random_state[0] << 11)
    random_state[0] = random_state[1]
    random_state[1] = random_state[2]
    random_state[2] = random_state[3]
    random_state[3] = random_state[3] ^ (random_state[3] >> 19) ^ t ^ (t >> 8)
    return random_state[3]


@numba.autojit(nopython=True, nogil=True)
def swap_1d(array_a, array_b):
    """
    Swap two arrays.

    :param array_a: array to swap to/from
    :param array_b: array to swap to/from
    """
    for x in range(array_a.shape[0]):
        array_a[x], array_b[x] = array_b[x], array_a[x]


@numba.autojit(nopython=True, nogil=True)
def partition_population(population, first_idx, last_idx):
    """
    The partition method from quicksort, modified to copy entire rows of
    population.  Sorts by first element of each row (the score).
    See: http://wki.pe/Quicksort

    :param population: 2D matrix, rows are members with score as first element
    :param first_idx: index of the first element to sort
    :param last_idx: index of the last element to sort
    :return: resulting pivot index
    """
    pivot_idx = (last_idx - first_idx) // 2 + first_idx
    pivot = population[pivot_idx, 0]
    swap_1d(population[pivot_idx], population[first_idx])

    low = first_idx + 1
    high = last_idx
    done = False
    while not done:
        while low <= high and population[low, 0] <= pivot:
            low += 1
        while population[high, 0] >= pivot and high >= low:
            high -= 1
        if high < low:
            done = True
        else:
            swap_1d(population[low], population[high])
    swap_1d(population[first_idx], population[high])
    return high


@numba.jit('void(uint32[:,:], uint32)', nopython=True, nogil=True)
def divide_population(population, divide_idx):
    """
    Split the population at divide_idx such that all members before divide_idx
    are better than all members after divide_idx.  Only partially sorts
    the population.

    :param population: 2D matrix, rows are members with score as first element
    :param divide_idx: index to divide population
    """
    first_idx = 0
    last_idx = population.shape[0] - 1
    split_idx = 0
    while first_idx < last_idx and split_idx != divide_idx:
        split_idx = partition_population(population, first_idx, last_idx)
        if split_idx > divide_idx:
            last_idx = split_idx - 1
        elif split_idx < divide_idx:
            first_idx = split_idx + 1
    # assert (population[:divide_idx, 0].max() <=
    #         population[divide_idx:, 0].min())


@numba.autojit(nopython=True, nogil=True)
def split_point(point):
    """
    Points are stored in a uint32 such that upper bits are i, lower bits are j.
    This method breaks an encoded point into i and j.

    :param point: uint32 point representation
    :return: (i, j) tuple
    """
    return point >> 16, point & 0xffff


@numba.autojit(nopython=True, nogil=True)
def shift_to(src_idx, dst_idx, array):
    """
    In-place shift.

    :param src_idx: index of element to shift
    :param dst_idx: index where element at src_idx should end up
    :param array: input/output array to do shifting on
    """
    val = array[src_idx]
    if src_idx < dst_idx:
        for i in range(src_idx, dst_idx):
            array[i] = array[i + 1]
    elif dst_idx < src_idx:
        i = src_idx
        while i > dst_idx:
            array[i] = array[i - 1]
            i -= 1
    array[dst_idx] = val


@numba.autojit(nopython=True, nogil=True)
def copy_1d(src, dst):
    """
    Copy all elements from array `src` to array `dst`
    """
    for i in range(dst.shape[0]):
        dst[i] = src[i]


@numba.autojit(nopython=True, nogil=True)
def copy_2d(src, dst):
    """
    Copy all elements from 2D matrix `src` to array `dst`
    """
    for k in range(dst.shape[0]):
        copy_1d(src[k], dst[k])


@numba.autojit(nopython=True, nogil=True)
def zero_fill_2d(dst):
    """
    Write `0` to every element in 2D matrix `dst`
    """
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i, j] = 0
