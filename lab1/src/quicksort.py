import random


def partition(array, start, end):
    rand_idx = random.randint(start, end)
    array[start], array[rand_idx] = array[rand_idx], array[start]

    pivot = array[start]
    low = start + 1
    high = end

    while True:
        while low <= high and array[high] >= pivot:
            high = high - 1

        while low <= high and array[low] <= pivot:
            low = low + 1

        if low <= high:
            array[low], array[high] = array[high], array[low]
        else:
            break

    array[start], array[high] = array[high], array[start]

    return high


def quick_sort(array, start=None, end=None):
    start = start if start is not None else 0
    end = end if end is not None else len(array) - 1

    if start >= end:
        return

    p = partition(array, start, end)
    quick_sort(array, start, p - 1)
    quick_sort(array, p + 1, end)
