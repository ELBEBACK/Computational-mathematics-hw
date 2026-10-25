import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 100
a = 10
b = lambda i, j: 1 / i
fi = lambda i: i

A = np.zeros((N, N))
f = np.zeros(N)
for i in range(1, N + 1):
    for j in range(1, min(i + 1, N) + 1):  # b_ij левее диагонали и b_i,i+1
        A[i-1, j-1] = a if j == i else b(i, j)
    f[i-1] = fi(i)

TOL = 1e-14
MAXIT = 1000



def gauss(A, f):
    M = np.column_stack([A, f])
    n = len(f)
    for k in range(n):
        p = k + np.argmax(np.abs(M[k:, k]))
        M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            M[i, k:] -= M[i, k] / M[k, k] * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i+1:n] @ x[i+1:]) / M[i, i]
    return x

def lu(A):
    n = len(A)
    L, U = np.eye(n), np.zeros((n, n))
    for i in range(n):
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]
        L[i+1:, i] = (A[i+1:, i] - L[i+1:, :i] @ U[:i, i]) / U[i, i]
    return L, U

def lu_solve(L, U, f):
    n = len(f)
    y = np.zeros(n)
    for i in range(n):
        y[i] = f[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


# Iterational ones

def jacobi(A, f, x):
    d = np.diag(A)
    while True:
        x = (f - (A @ x - d*x)) / d
        yield x

def sor(A, f, x, w):
    n = len(f)
    while True:
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i+1:] @ x[i+1:]
            x[i] = (1 - w)*x[i] + w*(f[i] - s) / A[i, i]
        yield x.copy()

def seidel(A, f, x):
    return sor(A, f, x, 1.0)

def steepest(A, f, x):
    while True:
        r = f - A @ x
        x = x + (r @ r) / (A @ r @ r) * r
        yield x

def minres(A, f, x):
    while True:
        r = f - A @ x
        Ar = A @ r
        x = x + (Ar @ r) / (Ar @ Ar) * r
        yield x

def cg(A, f, x):
    r = f - A @ x
    z = r.copy()
    while True:
        Az = A @ z
        alpha = (r @ r) / (Az @ z)
        x = x + alpha*z
        r1 = r - alpha*Az
        z = r1 + (r1 @ r1) / (r @ r) * z
        r = r1
        yield x

def cgnr(A, f, x):
    yield from cg(A.T @ A, A.T @ f, x)

def bicg(A, f, x):
    r = f - A @ x
    p, z, s = r.copy(), r.copy(), r.copy()
    while True:
        Az = A @ z
        alpha = (p @ r) / (s @ Az)
        x = x + alpha*z
        r1 = r - alpha*Az
        p1 = p - alpha*(A.T @ s)
        beta = (p1 @ r1) / (p @ r)
        z = r1 + beta*z
        s = p1 + beta*s
        r, p = r1, p1
        yield x

def bicgstab(A, f, x):
    r = f - A @ x
    rt = r.copy()
    rho = alpha = w = 1.0
    v = p = np.zeros_like(r)
    while True:
        rho1 = rt @ r
        beta = rho1 / rho * alpha / w
        p = r + beta*(p - w*v)
        v = A @ p
        alpha = rho1 / (rt @ v)
        s = r - alpha*v
        t = A @ s
        w = (t @ s) / (t @ t)
        x = x + alpha*p + w*s
        r = s - w*t
        rho = rho1
        yield x

def run(method):
    x = np.zeros(N)
    xs = [x]
    for x in method(A, f, x.copy()):
        xs.append(x)
        if np.linalg.norm(f - A @ x) / np.linalg.norm(f) < TOL or len(xs) > MAXIT:
            break
    return np.array(xs)


print("симметричная:", np.allclose(A, A.T))
print("строгое диаг. преобладание:", np.all(2*np.diag(A) > np.abs(A).sum(1)))
print(f"cond_2(A) = {np.linalg.cond(A):.4f}")

exact = np.linalg.solve(A, f)

def report(title, x, k="-"):
    res = np.linalg.norm(f - A @ x) / np.linalg.norm(f)
    err = np.linalg.norm(x - exact) / np.linalg.norm(exact)
    print(f"{title:<20} {k:>5} {res:>12.2e} {err:>12.2e}")

print(f"\n{'метод':<20} {'итер.':>5} {'||r||/||f||':>12} {'||x-x*||/||x*||':>12}")
report("Гаусс", gauss(A, f))
report("LU", lu_solve(*lu(A), f))

w = min(np.arange(1.01, 1.5, 0.01), key=lambda w: len(run(lambda A, f, x: sor(A, f, x, w))))

METHODS = [
    ("Якоби",              jacobi),
    ("Зейдель",            seidel),
    (f"SOR, ω={w:.2f}",    lambda A, f, x: sor(A, f, x, w)),
    ("градиентный спуск",  steepest),
    ("мин. невязки",       minres),
    ("CG",                 cg),
    ("CGNR (CG для AᵀA)",  cgnr),
    ("BiCG",               bicg),
    ("BiCGStab",           bicgstab),
]

results = []
for title, method in METHODS:
    xs = run(method)
    report(title, xs[-1], len(xs) - 1)
    results.append((title, xs))

print("\nx =", np.array2string(exact[:5], precision=6), "...",
      np.array2string(exact[-3:], precision=6))



plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#fbfbf8",
    "figure.facecolor": "white",
    "legend.frameon": False,
})

STYLE = [("#1f77b4", "o"), ("#ff7f0e", "s"), ("#2ca02c", "^"),
         ("#d62728", "D"), ("#9467bd", "v"), ("#8c564b", "P"),
         ("#e377c2", "X"), ("#7f7f7f", "*"), ("#17becf", "h")]

PLOTS = [
    ("resid.png", "Убывание невязки", r"$\|f - Ax_k\|_2 \,/\, \|f\|_2$",
     lambda xs: np.linalg.norm(f - xs @ A.T, axis=1) / np.linalg.norm(f)),
    ("error.png", "Убывание ошибки", r"$\|x_k - x^*\|_2 \,/\, \|x^*\|_2$",
     lambda xs: np.linalg.norm(xs - exact, axis=1) / np.linalg.norm(exact)),
]

for path, title, ylabel, norm in PLOTS:
    fig, ax = plt.subplots(figsize=(9, 6))
    for (label, xs), (col, mk) in zip(results, STYLE):
        ax.semilogy(norm(xs), marker=mk, color=col, lw=1.6, ms=5,
                    mfc="white", mew=1.3, label=label)
    ax.set_title(f"{title}, вариант е ($n={N}$, $a={a}$)", pad=12)
    ax.set_xlabel("итерация $k$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#d0d0d0", lw=0.7)
    ax.grid(True, which="minor", color="#ececec", lw=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="метод")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
