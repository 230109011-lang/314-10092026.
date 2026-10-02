"""
OpenMP Multi-Core Scaling in Python - ALL-IN-ONE RUNNER
Запуск:  python run_lab.py
Нужно:   pip install numpy numba matplotlib
Результат: таблица в консоли + results.txt + mandelbrot_output.png
"""
import time
import platform
import numpy as np
import numba
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")  # без окон, только сохранение в файл
import matplotlib.pyplot as plt

MAX_THREADS = numba.config.NUMBA_NUM_THREADS
LOG = []


def out(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- Challenge 1
@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples


def challenge1():
    out("=" * 60)
    out("CHALLENGE 1: Monte Carlo Pi")
    out("=" * 60)
    _ = monte_carlo_pi(10_000)  # warmup
    samples = 120_000_000
    counts = sorted(set(t for t in [1, 2, 4, 8, MAX_THREADS] if t <= MAX_THREADS))
    out(f"{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
    out("-" * 55)
    t1 = None
    rows = []
    for t in counts:
        numba.set_num_threads(t)
        s = time.perf_counter()
        monte_carlo_pi(samples)
        el = time.perf_counter() - s
        if t == 1:
            t1 = el
        sp = t1 / el
        eff = sp / t * 100.0
        rows.append((t, el, sp, eff))
        out(f"{t:<10} | {el:<12.4f} | {sp:<10.2f}x | {eff:<15.1f}")
    numba.set_num_threads(MAX_THREADS)
    return rows


# ---------------------------------------------------------------- Challenge 2
@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


def challenge2():
    out("")
    out("=" * 60)
    out("CHALLENGE 2: Mandelbrot (load imbalance)")
    out("=" * 60)
    numba.set_num_threads(MAX_THREADS)
    _ = render_mandelbrot_rows(100, 100, 50)
    _ = render_mandelbrot_cols(100, 100, 50)
    H, W, MAX_IT = 2500, 2500, 1000

    t0 = time.perf_counter()
    grid_rows = render_mandelbrot_rows(H, W, MAX_IT)
    t_rows = time.perf_counter() - t0

    t1 = time.perf_counter()
    render_mandelbrot_cols(H, W, MAX_IT)
    t_cols = time.perf_counter() - t1

    out(f"Row-Parallel Render Time:    {t_rows:.3f} s")
    out(f"Column-Parallel Render Time: {t_cols:.3f} s")
    faster = "Rows" if t_rows < t_cols else "Columns"
    out(f"Faster decomposition: {faster}")

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_rows, cmap="magma", extent=[-2.0, 0.5, -1.2, 1.2])
    plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
    plt.axis("off")
    plt.savefig("mandelbrot_output.png", dpi=300, bbox_inches="tight")
    plt.close()
    out("Saved image: mandelbrot_output.png")
    return t_rows, t_cols


# ---------------------------------------------------------------- Challenge 3
@njit(parallel=True)
def heat_step(u, u_next, alpha):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4.0 * u[i, j]
            )


def run_heat(dtype, grid=1500, steps=300):
    u = np.zeros((grid, grid), dtype=dtype)
    u_next = np.zeros_like(u)
    for a in (u, u_next):
        a[0, :] = 100.0
        a[:, 0] = 100.0
    alpha = dtype(0.20)
    heat_step(u, u_next, alpha)  # warmup / compile for this dtype
    s = time.perf_counter()
    for _ in range(steps):
        heat_step(u, u_next, alpha)
        u, u_next = u_next, u
    el = time.perf_counter() - s
    mcells = (grid * grid * steps) / el / 1e6
    return el, mcells


def challenge3():
    out("")
    out("=" * 60)
    out("CHALLENGE 3: Heat stencil (memory bandwidth)")
    out("=" * 60)
    numba.set_num_threads(MAX_THREADS)
    e64, m64 = run_heat(np.float64)
    out(f"float64: {e64:.3f} s | {m64:.2f} Megacells/sec")
    e32, m32 = run_heat(np.float32)
    out(f"float32: {e32:.3f} s | {m32:.2f} Megacells/sec")
    out(f"float32 faster by factor: {e64 / e32:.2f}x")
    return e64, m64, e32, m32


# ---------------------------------------------------------------- Main
def main():
    out(f"CPU: {platform.processor() or platform.machine()} | {platform.system()}")
    out(f"Hardware Threads Detected: {MAX_THREADS}")
    out("(Ноутбук должен быть в розетке, закрой браузер и прочие программы)")
    out("")

    c1 = challenge1()
    t_rows, t_cols = challenge2()
    e64, m64, e32, m32 = challenge3()

    t1 = c1[0][1]
    out("")
    out("=" * 60)
    out("TABLE 1 - SCORECARD")
    out("=" * 60)
    out(f"{'Benchmark':<28}{'Threads':<9}{'Time (s)':<11}{'Speedup':<10}{'Eff (%)':<8}")
    for t, el, sp, eff in c1:
        label = "1 (Base)" if t == 1 else str(t)
        out(f"{'C1 Monte Carlo':<28}{label:<9}{el:<11.4f}{sp:<10.2f}{eff:<8.1f}")
    out(f"{'C2 Mandelbrot (Rows)':<28}{MAX_THREADS:<9}{t_rows:<11.3f}{'N/A':<10}{'N/A':<8}")
    out(f"{'C2 Mandelbrot (Cols)':<28}{MAX_THREADS:<9}{t_cols:<11.3f}{'N/A':<10}{'N/A':<8}")
    out(f"{'C3 Heat Stencil (f64)':<28}{MAX_THREADS:<9}{e64:<11.3f}{'N/A':<10}{'N/A':<8}")
    out(f"{'C3 Heat Stencil (f32)':<28}{MAX_THREADS:<9}{e32:<11.3f}{'N/A':<10}{'N/A':<8}")
    out("")
    out(f"T_1 (1 thread) = {t1:.4f} s | T_max ({c1[-1][0]} threads) = {c1[-1][1]:.4f} s")
    out(f"WHITEBOARD -> Best Monte Carlo speedup: {max(r[2] for r in c1):.2f}x")
    out(f"C3 throughput (float64): {m64:.2f} Megacells/sec")

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    out("\nSaved: results.txt, mandelbrot_output.png")


if __name__ == "__main__":
    main()
