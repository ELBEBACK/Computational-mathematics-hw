import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X0 = 6.7

FUNCS = [
    (r"$\sin(x^2)$",
     lambda x: np.sin(x**2),
     lambda x: 2*x*np.cos(x**2)),
    (r"$\cos(\sin x)$",
     lambda x: np.cos(np.sin(x)),
     lambda x: -np.sin(np.sin(x))*np.cos(x)),
    (r"$\exp(\sin(\cos x))$",
     lambda x: np.exp(np.sin(np.cos(x))),
     lambda x: -np.exp(np.sin(np.cos(x)))*np.cos(np.cos(x))*np.sin(x)),
    (r"$\ln(x+3)$",
     lambda x: np.log(x + 3),
     lambda x: 1/(x + 3)),
    (r"$(x+3)^{0.5}$",
     lambda x: np.sqrt(x + 3),
     lambda x: 0.5/np.sqrt(x + 3)),
]

def c(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

METHODS = [
    ("1: вперёд, $O(h)$",       lambda f, x, h: (f(x + h) - f(x)) / h),     # вправо, вдоль оси #
    ("2: назад, $O(h)$",        lambda f, x, h: (f(x) - f(x - h)) / h),     # влево, против оси #   
    ("3: центр., $O(h^2)$",     lambda f, x, h: c(f, x, h)),                # с двух сторон #
    ("4: $O(h^4)$",             lambda f, x, h: 4/3*c(f, x, h) - 1/3*c(f, x, 2*h)),
    ("5: $O(h^6)$",             lambda f, x, h: 3/2*c(f, x, h) - 3/5*c(f, x, 2*h)
                                               + 1/10*c(f, x, 3*h)),
]

n = np.arange(1, 22)
h = 2.0 / 2.0**n
EPS = np.finfo(float).eps

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

STYLE = [("#1f77b4", "o", 1), ("#ff7f0e", "s", 1), ("#2ca02c", "^", 2),
         ("#d62728", "D", 4), ("#9467bd", "v", 6)]

for k, (name, f, df) in enumerate(FUNCS, 1):
    exact = df(X0)
    fig, ax = plt.subplots(figsize=(9, 6))
    top = 0
    for (label, method), (col, mk, p) in zip(METHODS, STYLE):
        err = np.array([abs(method(f, X0, hi) - exact) for hi in h])
        err = np.where(err > 0, err, np.nan)
        top = max(top, np.nanmax(err))
        ax.loglog(h, err, marker=mk, color=col, lw=1.6, ms=5,
                  mfc="white", mew=1.3, label=label)
        ok = (h <= 1/4) & (err > 1e2 * EPS * abs(f(X0)) / h)
        if ok.any():
            C = np.exp(np.mean(np.log(err[ok]) - p * np.log(h[ok])))
            ax.loglog(h, C * h**p, ls="--", lw=0.8, color=col, alpha=0.5)
    ax.loglog(h, EPS * abs(f(X0)) / h, ls=":", lw=1.5, color="gray",
              label=r"округление $\varepsilon|f|/h$")
    ax.plot([], [], ls="--", color="gray", lw=0.8, label=r"$\propto h^p$")
    ax.set_ylim(1e-17, 10 * top)
    ax.set_title(f"Погрешность производной {name} в точке $x_0={X0}$", pad=12)
    ax.set_xlabel("шаг $h$")
    ax.set_ylabel(r"$|f'_{\mathrm{num}}(x_0) - f'(x_0)|$")
    ax.grid(True, which="major", color="#d0d0d0", lw=0.7)
    ax.grid(True, which="minor", color="#ececec", lw=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="метод")
    fig.tight_layout()
    fig.savefig(f"err_{k}.png", dpi=130)
    plt.close(fig)
