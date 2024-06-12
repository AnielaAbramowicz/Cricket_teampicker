import numpy as np
import timeit


def divide_with_np(l1, l2):
    l2 = l2.astype(float)
    l1 = l1.astype(float)
    np.divide(l1, l2, out=np.zeros_like(l1), where=l2 != 0)

def divide_with_zip(l1, l2):
    [j / i if i != 0 else 0 for i, j in zip(l1, l2)]


def main():
    l1 = np.random.randint(1, 100, 1000)
    l2 = np.random.randint(1, 100, 1000)
    iters = 10000
    print('Numpy divide')
    print(timeit.timeit(lambda: divide_with_np(l1, l2), number=iters))
    print('Zip divide')
    print(timeit.timeit(lambda: divide_with_zip(l1, l2), number=iters))

if __name__ == '__main__':
    main()

