import numpy as np
import time
from numpy.ctypeslib import ndpointer
import ctypes
from mkl import mklblas

# Размер матрицы
N = 1024

np.random.seed(42)
A = np.random.randn(N, N).astype(np.complex64) + 1j * np.random.randn(N, N).astype(np.complex64)
B = np.random.randn(N, N).astype(np.complex64) + 1j * np.random.randn(N, N).astype(np.complex64)

c = 2 * N**3
print(f"Теоретическая сложность: {c} операций")

# 1 Прямой алгоритм
def direct_matrix_mult(a, b):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.complex64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i,j] += a[i,k] * b[k,j]
    return c

start = time.time()
C_direct = direct_matrix_mult(A, B)
t_direct = time.time() - start
p_direct = c / t_direct * 1e-6
print(f"Прямой алгоритм: {t_direct:.3f} сек, {p_direct:.2f} MFlops")

# 2 BLAS через MKL
def blas_matrix_mult(a, b):
    c = np.zeros((N, N), dtype=np.complex64)
    mklblas.cgemm(a, b, c, transa='N', transb='N', alpha=1.0, beta=0.0)
    return c

start = time.time()
C_blas = blas_matrix_mult(A, B)
t_blas = time.time() - start
p_blas = c / t_blas * 1e-6
print(f"BLAS (cgemm): {t_blas:.3f} сек, {p_blas:.2f} MFlops")

# 3 Оптимизированный алгоритм (блочное умножение)
def blocked_matrix_mult(a, b, block_size=64):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.complex64)
    
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii+block_size, n)):
                    for j in range(jj, min(jj+block_size, n)):
                        temp = 0
                        for k in range(kk, min(kk+block_size, n)):
                            temp += a[i,k] * b[k,j]
                        c[i,j] += temp
    return c

start = time.time()
C_blocked = blocked_matrix_mult(A, B)
t_blocked = time.time() - start
p_blocked = c / t_blocked * 1e-6
print(f"Блочный алгоритм: {t_blocked:.3f} сек, {p_blocked:.2f} MFlops")

# Проверка корректности
assert np.allclose(C_direct, C_blas, atol=1e-3)
assert np.allclose(C_direct, C_blocked, atol=1e-3)
