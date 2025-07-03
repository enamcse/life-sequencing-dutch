import time
import math
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def mydist(x, y, **kwargs):
    f = kwargs["f"]
    return np.sum(np.abs(x - y) ** f) ** (1. / f)


def nearest_nei_dist_mean(X, k, distance='cosine', f=None):
    if distance == 'cosine':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='cosine', include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = 1. - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    aver_dist.append(dist)
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'l2_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=2, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.linalg.norm(x - np.asarray(X[j])))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'l1_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=1, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j]))))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'frac_distance':
        p = len(X[0])
        if f is None:
            f = 1. / p
        A = kneighbors_graph(X=X, n_neighbors=k, metric=mydist, metric_params={"f": f}, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j])) ** f) ** (1. / f))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list


def nearest_nei_dist_std(X, k, distance='cosine'):
    if distance == 'cosine':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='cosine', include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = 1. - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    aver_dist.append(dist)
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list
    elif distance == 'l2_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=2, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = np.linalg.norm(x - y)
                    aver_dist.append(dist)
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list
    elif distance == 'l1_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=1, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j]))))
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list


def int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


def primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10, ) -> float:
    """Returns func inverse of target between lower and upper
    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def uniform_sphere(d: int, n: int) -> List[List[float]]:
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points


def gaussian_sphere(n: int, d: int = 3) -> List[List[float]]:
    points = [[] for _ in range(n)]
    for i in range(n):
        x = np.random.normal(0.0, 1.0, d)
        x = x / np.linalg.norm(x)
        points[i] = list(x)
    return points


def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return [list(_) for _ in points]


def test():
    n = 1000
    d = 3

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    gau_points = gaussian_sphere(n=n, d=d)
    gau_points = np.asarray(gau_points)
    ax.scatter3D(gau_points[:, 0], gau_points[:, 1], gau_points[:, 2])
    ax.set_title(f'Gaussian-Points(n={n})')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    equ_points = uniform_sphere(n=n, d=d)
    equ_points = np.asarray(equ_points)
    ax.scatter3D(equ_points[:, 0], equ_points[:, 1], equ_points[:, 2])
    ax.set_title(f'Equal-Points(n={n})')
    plt.show()
    # consider averaged distance
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(n=1000, d=3)
        aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
        X2 = uniform_sphere(n=1000, d=3)
        aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
        ax[ind].legend()
    plt.show()


def test_1():
    n = 2000

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    ax.set_title(f'General-Fibonacci(n={n})')
    plt.show()


def test_2():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(2000)
        aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
        X2 = fibonacci_sphere(2000)
        aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
        ax[ind].legend()
    plt.show()


def test_3():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(n=1000, d=3)
        aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
        X2 = uniform_sphere(n=1000, d=3)
        aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
        ax[ind].legend()
    plt.show()


def test_4():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(10000)
        aver_dist_X1 = nearest_nei_dist_std(X1, k=5, distance=dist)
        X2 = fibonacci_sphere(10000)
        aver_dist_X2 = nearest_nei_dist_std(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
        ax[ind].legend()
    plt.show()


def improved(n):
    if n >= 600000:
        epsilon = 214
    elif n >= 400000:
        epsilon = 75
    elif n >= 11000:
        epsilon = 27
    elif n >= 890:
        epsilon = 10
    elif n >= 177:
        epsilon = 3.33
    elif n >= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + epsilon) / (n - 1 + 2 * epsilon))
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return [[x[_], y[_], z[_]] for _ in range(len(x))]


def test_6():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(10000)
        aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
        X2 = improved(10000)
        aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='Improved', alpha=.5)
    plt.show()


def test_7():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
        X1 = gaussian_sphere(10000)
        aver_dist_X1 = nearest_nei_dist_std(X1, k=5, distance=dist)
        X2 = improved(10000)
        aver_dist_X2 = nearest_nei_dist_std(X2, k=5, distance=dist)
        ax[ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
        ax[ind].hist(aver_dist_X2, bins=50, color='g', label='Improved', alpha=.5)
    plt.show()


def test_8():
    fig, ax = plt.subplots(5, 3, figsize=(18, 20))
    for j, d in enumerate([3, 4, 5, 6, 7]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
            X1 = gaussian_sphere(n=5000, d=d)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
            X2 = uniform_sphere(n=5000, d=d)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
    plt.show()

    fig, ax = plt.subplots(5, 3, figsize=(18, 20))
    for j, d in enumerate([8, 9, 10, 11, 12]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
            X1 = gaussian_sphere(n=5000, d=d)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
            X2 = uniform_sphere(n=5000, d=d)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
    plt.show()

    fig, ax = plt.subplots(5, 3, figsize=(18, 20))
    for j, n in enumerate([1000, 3000, 5000, 7000, 10000]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance']):
            X1 = uniform_sphere(n=n, d=16)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist)
            X2 = gaussian_sphere(n=n, d=16)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for j, n in enumerate([1000, 2000]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance', 'frac_distance']):
            X1 = gaussian_sphere(n=n, d=16)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist, f=None)
            X2 = uniform_sphere(n=n, d=16)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist, f=None)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
            ax[j][ind].legend()
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for j, n in enumerate([1000, 2000]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance', 'frac_distance']):
            X1 = gaussian_sphere(n=n, d=16)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist, f=0.1)
            X2 = uniform_sphere(n=n, d=16)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist, f=0.1)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
            ax[j][ind].legend()
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for j, n in enumerate([1000, 2000]):
        for ind, dist in enumerate(['cosine', 'l2_distance', 'l1_distance', 'frac_distance']):
            X1 = gaussian_sphere(n=n, d=16)
            aver_dist_X1 = nearest_nei_dist_mean(X1, k=5, distance=dist, f=0.6)
            X2 = uniform_sphere(n=n, d=16)
            aver_dist_X2 = nearest_nei_dist_mean(X2, k=5, distance=dist, f=0.6)
            ax[j][ind].hist(aver_dist_X1, bins=50, color='b', label='Gaussian', alpha=.5)
            ax[j][ind].hist(aver_dist_X2, bins=50, color='g', label='GenFibonacci', alpha=.5)
            ax[j][ind].legend()
    plt.show()


def test_9():
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))

    k = 5
    n = 100
    d_list = range(3, 32, 2)
    dist_list = ['cosine', 'l2_distance', 'l1_distance', 'frac_distance']
    dist_by_dim_uniform = np.zeros((len(dist_list), len(d_list)))
    dist_by_dim_gaussian = np.zeros((len(dist_list), len(d_list)))
    for i, d in enumerate(d_list):
        xx = uniform_sphere(d=d, n=n)
        yy = gaussian_sphere(d=d, n=n)
        for j, dist in enumerate(dist_list):
            dist_by_dim_uniform[j][i] = np.mean(nearest_nei_dist_mean(xx, k=k, distance=dist, f=0.6))
            dist_by_dim_gaussian[j][i] = np.mean(nearest_nei_dist_mean(yy, k=k, distance=dist, f=0.6))
    for i in range(4):
        ax[i].plot(d_list, dist_by_dim_uniform[i], label='General-Fibonacci')
        ax[i].plot(d_list, dist_by_dim_gaussian[i], label='Gaussian')
        ax[i].legend()
        ax[i].set_xlim([3, 32])
        ax[i].set_ylabel(f'Minimum average distance k=({k})')
        ax[i].set_xlabel(f'Dimension d')
        ax[i].set_title(f'n={n}, k={k}, dist_metric={dist_list[i]}')
    plt.show()


def test_10():
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))

    k = 5
    n = 1000
    d_list = range(3, 32, 2)
    dist_list = ['cosine', 'l2_distance', 'l1_distance', 'frac_distance']
    dist_by_dim_uniform = np.zeros((len(dist_list), len(d_list)))
    dist_by_dim_gaussian = np.zeros((len(dist_list), len(d_list)))
    for i, d in enumerate(d_list):
        xx = uniform_sphere(d=d, n=n)
        yy = gaussian_sphere(d=d, n=n)
        for j, dist in enumerate(dist_list):
            dist_by_dim_uniform[j][i] = np.mean(nearest_nei_dist_mean(xx, k=k, distance=dist, f=0.6))
            dist_by_dim_gaussian[j][i] = np.mean(nearest_nei_dist_mean(yy, k=k, distance=dist, f=0.6))
    for i in range(4):
        ax[i].plot(d_list, dist_by_dim_uniform[i], label='General-Fibonacci')
        ax[i].plot(d_list, dist_by_dim_gaussian[i], label='Gaussian')
        ax[i].legend()
        ax[i].set_xlim([3, 32])
        ax[i].set_ylabel(f'Minimum average distance k=({k})')
        ax[i].set_xlabel(f'Dimension d')
        ax[i].set_title(f'n={n}, k={k}, dist_metric={dist_list[i]}')
    plt.show()

def pairwise_distances(points):
    """
    Compute pairwise L2 distances for a list of points in d dimensions.

    Parameters:
        points (list or np.ndarray): A list or NumPy array of shape (n, d),
                                     where n is the number of points and d is the dimensionality.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n, n) containing pairwise L2 distances.
    """
    points = np.asarray(points)
    # Compute the squared differences and sum them row-wise for the pairwise distances
    pairwise_sq_dists = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
    # Take the square root to compute L2 distances
    return np.sqrt(pairwise_sq_dists)

p = uniform_sphere(4, 3)
print(pairwise_distances(p))