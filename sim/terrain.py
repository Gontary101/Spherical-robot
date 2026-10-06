"""
Procedural terrain for GYRA Mk2 training and evaluation (MuJoCo heightfield).

Every generator returns a height map in metres on a square grid (`N x N` cells, `CELL` m each),
centred on the origin. The map is written into a MuJoCo `hfield`, which collides with the two tyre
spheres exactly (sphere-prism), so what the robot drives on is what the generator produced.

Families (difficulty `k` in [0, 1] scales the amplitude):
  flat    : nothing (friction / slope / disturbances still vary)
  hills   : band-limited noise, 1.5-6 m wavelengths (p95 slope ~12 deg at k=1)
  rough   : hills + gravel/potholes (0.3-0.6 m wavelength, ~3 cm)
  curbs   : random straight curbs / steps up to 6 cm and shallow ditches
  ramps   : planar ramps up to 15 deg with flat landings
  mixed   : patches of all of the above
"""
import numpy as np

CELL = 0.15          # m
N = 400              # cells per side -> 60 m square
HALF = N * CELL / 2
FAMILIES = ("flat", "hills", "rough", "curbs", "ramps", "mixed")


def _band_noise(rng, lo_wl, hi_wl, n=None):
    """Gaussian noise band-passed to wavelengths in [lo_wl, hi_wl] metres, unit std."""
    n = n or N
    f = np.fft.fftfreq(n, d=CELL)
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    mask = (fr >= 1.0 / hi_wl) & (fr <= 1.0 / lo_wl)
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * mask
    h = np.real(np.fft.ifft2(spec))
    s = h.std()
    return h / s if s > 0 else h


def _grid():
    """World (X, Y) of every heightfield VERTEX, exactly as MuJoCo lays them out: x = -HALF + c * 2 HALF / (N - 1)."""
    x = np.linspace(-HALF, HALF, N)
    return np.meshgrid(x, x)          # X varies along columns, Y along rows


def hills(rng, k):
    return 0.045 * k * _band_noise(rng, 1.5, 6.0)


def rough(rng, k):
    return hills(rng, 0.7 * k) + 0.012 * k * _band_noise(rng, 0.3, 0.6)


def curbs(rng, k):
    X, Y = _grid()
    h = np.zeros((N, N))
    for _ in range(int(rng.integers(25, 45))):
        th = rng.uniform(0, np.pi)
        c = rng.uniform(-HALF, HALF, 2)
        u = (X - c[0]) * np.cos(th) + (Y - c[1]) * np.sin(th)
        v = -(X - c[0]) * np.sin(th) + (Y - c[1]) * np.cos(th)
        length, width = rng.uniform(2, 10), rng.uniform(0.3, 3.0)
        band = (np.abs(v) < length / 2) & (u > 0) & (u < width)
        step = rng.uniform(0.01, 0.06) * k * (1 if rng.random() < 0.75 else -1)
        h[band] += step
    return h + 0.004 * k * _band_noise(rng, 0.3, 0.6)


def ramps(rng, k):
    X, Y = _grid()
    h = np.zeros((N, N))
    for _ in range(int(rng.integers(6, 12))):
        th = rng.uniform(-np.pi, np.pi)
        c = rng.uniform(-HALF + 3, HALF - 3, 2)
        u = (X - c[0]) * np.cos(th) + (Y - c[1]) * np.sin(th)
        v = -(X - c[0]) * np.sin(th) + (Y - c[1]) * np.cos(th)
        L, W = rng.uniform(1.5, 4), rng.uniform(1.5, 4)
        top = rng.uniform(1.0, 2.5)
        slope = np.tan(np.radians(rng.uniform(4, 15) * k + 1e-3))
        rise = np.clip(u, 0, L) * slope
        prof = np.where(u < L, rise, np.where(u < L + top, L * slope, np.maximum(L * slope - (u - L - top) * slope, 0)))
        prof = np.where(u < 0, 0, prof)
        side = np.maximum(W / 2 + L - np.abs(v), 0) * slope           # sides taper at the same grade: no cliffs
        h = np.maximum(h, np.minimum(prof, side))
    return h


GEN = dict(flat=lambda r, k: np.zeros((N, N)), hills=hills, rough=rough, curbs=curbs, ramps=ramps)


def mixed(rng, k):
    """Smoothly blended patches (soft-max over 6-20 m selector fields: no seams or cliffs between them)."""
    names = ("flat", "hills", "rough", "curbs", "ramps")
    sel = np.stack([_band_noise(rng, 6, 20) for _ in names])
    w = np.exp(3.0 * sel)
    w /= w.sum(0)
    return sum(w[i] * GEN[nm](rng, k) for i, nm in enumerate(names))


GEN["mixed"] = mixed


def make(rng, family, k):
    h = GEN[family](rng, k)
    # keep a flat 1.2 m pad at the centre (spawn) and fade to flat at the borders
    X, Y = _grid()
    r = np.hypot(X, Y)
    pad = np.clip((r - 0.6) / 1.2, 0, 1)
    edge = np.clip((HALF - np.maximum(np.abs(X), np.abs(Y))) / 2.0, 0, 1)
    h = h * pad * edge
    h -= h[N // 2, N // 2]
    return h


def height_at(h, x, y):
    """Height of the heightfield surface at world (x, y), identical to what MuJoCo collides with and ray-casts:
    vertices on the N x N grid spanning [-HALF, HALF], each cell split along its (r, c)-(r+1, c+1) diagonal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = 2 * HALF / (N - 1)
    c = np.clip((x + HALF) / sp, 0, N - 1 - 1e-9)
    r = np.clip((y + HALF) / sp, 0, N - 1 - 1e-9)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    u, v = c - c0, r - r0
    z00, z01, z10, z11 = h[r0, c0], h[r0, c0 + 1], h[r0 + 1, c0], h[r0 + 1, c0 + 1]
    upper = u >= v
    return np.where(upper, z00 + u * (z01 - z00) + v * (z11 - z01), z00 + v * (z10 - z00) + u * (z11 - z10))


def mesh(h):
    """Triangle mesh (vertices, faces) of the heightfield with MuJoCo's exact triangulation (for rendering)."""
    X, Y = _grid()
    V = np.c_[X.ravel(), Y.ravel(), h.ravel()]
    idx = np.arange(N * N).reshape(N, N)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    F = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])
    return V, F


def hfield_xml(h, name="terrain"):
    """(asset xml, geom xml, normalised data) for MuJoCo. Heights are stored as (h - lo) / (hi - lo)."""
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-4)
    data = ((h - lo) / span).astype(np.float32)
    asset = f'<hfield name="{name}" nrow="{N}" ncol="{N}" size="{HALF:.4f} {HALF:.4f} {span:.5f} 0.5"/>'
    return asset, lo, data
