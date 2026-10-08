"""Build (and optionally execute) validation.ipynb.

The notebook generates reference Python data structures, writes each of them
to every file format pyferrite supports that can hold it, records a manifest
of what Rust must find, and draws the reference figure (top row of Fig. 3).

    python make_notebook.py            # write validation.ipynb
    python make_notebook.py --execute  # write and run it in place
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# pyferrite: validation data

This notebook creates a set of typical Python data structures, saves each one in
every format the `pyferrite` crate supports that is able to hold it, and writes
`data/manifest.json` with what a correct reader must find: the path of every
value, its type, shape, checksum, first and last element.

The Rust project in `rust/` then reads every file with `pyferrite` from
crates.io, checks it against the manifest and draws the same three panels as
the figure at the end of this notebook.
""")

code("""
import json, pickle, platform, sys, warnings
from pathlib import Path

import h5py, joblib, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")
DATA, FIG = Path("data"), Path("figures")
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
for f in DATA.iterdir():
    f.unlink()
rng = np.random.default_rng(2026)
print("python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__,
      "| torch", torch.__version__, "| h5py", h5py.__version__, "| joblib", joblib.__version__)
""")

md("## 1. Manifest: what a correct reader must see")

code('''
MANIFEST = []          # one entry per (file, value path)

def np_name(a):
    k = a.dtype.kind
    return {"U": "str", "S": "bytes"}.get(k, a.dtype.name)

def leaves(obj, path=""):
    """Flatten a Python value the way the Rust checker flattens pyferrite::Value."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        tag = "list" if isinstance(obj, list) else "tuple"
        yield path, {"kind": tag, "len": len(obj)}
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, (set, frozenset)):
        yield path, {"kind": "set", "items": sorted(repr(x) for x in obj)}
    elif isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            yield from leaves(obj[c].to_numpy(), f"{path}/{c}" if path else str(c))
    elif isinstance(obj, np.ndarray):
        e = {"kind": "array", "dtype": np_name(obj), "shape": list(obj.shape)}
        if obj.dtype.kind in "biuf":
            flat = obj.astype(np.float64).ravel(order="C")
            e.update(sum=float(flat.sum()), first=float(flat[0]) if flat.size else None,
                     last=float(flat[-1]) if flat.size else None)
        elif obj.dtype.kind == "c":
            flat = obj.astype(np.complex128).ravel()
            e.update(sum=float(flat.real.sum() + flat.imag.sum()))
        elif obj.dtype.kind in "US":
            e.update(first=str(obj.ravel()[0]) if obj.dtype.kind == "U" else obj.ravel()[0].hex())
        yield path, e
    elif isinstance(obj, (bool, np.bool_)):
        yield path, {"kind": "bool", "value": bool(obj)}
    elif isinstance(obj, (int, np.integer)):
        yield path, {"kind": "int", "value": str(int(obj))}
    elif isinstance(obj, (float, np.floating)):
        yield path, {"kind": "float", "value": repr(float(obj))}
    elif isinstance(obj, complex):
        yield path, {"kind": "complex", "re": obj.real, "im": obj.imag}
    elif isinstance(obj, str):
        yield path, {"kind": "str", "value": obj}
    elif isinstance(obj, bytes):
        yield path, {"kind": "bytes", "hex": obj.hex()}
    elif obj is None:
        yield path, {"kind": "none"}
    else:
        raise TypeError(f"no manifest rule for {type(obj)} at {path}")

def record(fname, structure, obj, **extra):
    entries = [dict(path=p, **e) for p, e in leaves(obj)]
    MANIFEST.append(dict(file=fname, structure=structure, entries=entries, **extra))
''')

md("## 2. Writers: one per format")

code('''
def as_torch(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "biufc":
        # .copy() keeps 0-d arrays 0-d (np.ascontiguousarray would make them 1-d)
        return torch.from_numpy(obj.copy(order="C"))
    if isinstance(obj, dict):
        return {k: as_torch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [as_torch(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(as_torch(v) for v in obj)
    return obj

def flat_keys(obj, prefix=""):
    """npz has no nesting: join nested dict keys with '/', as pyferrite expects."""
    out = {}
    for k, v in obj.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flat_keys(v, key))
        else:
            out[key] = v
    return out

def h5_write(group, obj):
    for k, v in obj.items():
        if isinstance(v, dict):
            h5_write(group.create_group(k), v)
        else:
            group.create_dataset(k, data=v)

WRITERS = {
    "npy":    lambda p, o: np.save(p, o),
    "npz":    lambda p, o: np.savez(p, **flat_keys(o)),
    "npz_c":  lambda p, o: np.savez_compressed(p, **flat_keys(o)),
    "pkl":    lambda p, o: pickle.dump(o, open(p, "wb"), protocol=5),
    "pkl2":   lambda p, o: pickle.dump(o, open(p, "wb"), protocol=2),
    "joblib": lambda p, o: joblib.dump(o, p),
    "joblib_z": lambda p, o: joblib.dump(o, p, compress=3),
    "pt":     lambda p, o: torch.save(as_torch(o), p),
    "h5":     lambda p, o: h5_write(h5py.File(p, "w"), o),
}
EXT = {"npz_c": "npz", "pkl2": "pkl", "joblib_z": "joblib"}

def save(name, obj, formats, **extra):
    for fmt in formats:
        fname = f"{name}.{fmt}.{EXT.get(fmt, fmt)}" if fmt in EXT else f"{name}.{fmt}"
        WRITERS[fmt](DATA / fname, obj)
        record(fname, name, obj, **extra)
''')

md("## 3. The structures")

code('''
# --- 3.1 time series: two smooth curves with closed-form formulas -----------
t = np.linspace(0.0, 99.0, 100)                      # 100 points, step 1
series = {
    "t": t,
    "sin": np.sin(2 * np.pi * t / 50),               # two full periods
    "parabola": ((t - 49.5) / 49.5) ** 2 * 2 - 1,     # from 1 down to -1 and back
}
save("series", series, ["npz", "npz_c", "pkl", "pkl2", "joblib", "joblib_z", "pt", "h5"])
save("series_sin", series["sin"], ["npy"])

# the same series as a pandas DataFrame (pickled BlockManager)
frame = pd.DataFrame(series)
save("series_frame", frame, ["pkl", "joblib"])
''')

code('''
# --- 3.2 k-NN trained on Fisher's irises (PCA to two components) ------------
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X2 = make_pipeline(StandardScaler(), PCA(n_components=2)).fit_transform(iris.data)
knn = KNeighborsClassifier(n_neighbors=5).fit(X2, iris.target)
print("k-NN training accuracy:", knn.score(X2, iris.target))

knn_params = {
    "X": knn._fit_X.astype(np.float64),
    "y": knn._y.astype(np.int64),
    "classes": np.asarray(iris.target_names),        # unicode strings
    "n_neighbors": knn.n_neighbors,
}
save("knn_params", knn_params, ["pkl", "joblib", "joblib_z", "pt"])
knn_arrays = {k: v for k, v in knn_params.items() if isinstance(v, np.ndarray)}
save("knn_arrays", knn_arrays, ["npz"])
# h5py stores no numpy unicode strings, so the HDF5 copy keeps the numbers only
save("knn_arrays_h5", {k: v for k, v in knn_arrays.items() if v.dtype.kind != "U"}, ["h5"])
# the whole fitted estimator: an arbitrary sklearn class. pyferrite never runs
# it: the class is captured inertly and its fitted arrays stay readable.
joblib.dump(knn, DATA / "knn_estimator.joblib")
MANIFEST.append(dict(file="knn_estimator.joblib", structure="knn_estimator",
                     object="sklearn.neighbors._classification.KNeighborsClassifier",
                     entries=[dict(path=p, **e) for p, e in leaves(
                         {"_fit_X": knn._fit_X, "_y": knn._y})]))
''')

code('''
# --- 3.3 a 2-D field as a torch tensor: two Gaussian bumps -----------------
g = np.linspace(-3.0, 3.0, 61)
gx, gy = np.meshgrid(g, g)
field = (np.exp(-((gx - 1.0) ** 2 + gy ** 2))
         + 0.7 * np.exp(-((gx + 1.2) ** 2 + (gy - 1.0) ** 2) / 0.5)).astype(np.float32)
field_obj = {"field": field, "x": g.astype(np.float32), "y": g.astype(np.float32)}
save("field", field_obj, ["pt", "npz", "h5", "pkl", "joblib"])

# a small PyTorch network: its state_dict, as torch.save writes it
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
torch.save(net.state_dict(), DATA / "torch_mlp.pt")
record("torch_mlp.pt", "torch_mlp", sd)
''')

code('''
# --- 3.4 a TensorFlow / Keras model fitted to the sine series ---------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.random.set_seed(0)
model = tf.keras.Sequential([tf.keras.layers.Input((1,)),
                             tf.keras.layers.Dense(16, activation="tanh", name="hidden"),
                             tf.keras.layers.Dense(1, name="out")])
model.compile("adam", "mse")
model.fit(t[:, None] / 99.0, series["sin"], epochs=50, verbose=0)
tf_weights = {f"{layer.name}/{w.name.split('/')[-1].split(':')[0]}": w.numpy()
              for layer in model.layers for w in layer.weights}
tf_nested = {}
for k, v in tf_weights.items():
    a, b = k.split("/"); tf_nested.setdefault(a, {})[b] = v
save("tf_dense", tf_nested, ["npz", "h5", "pkl", "pt"])
print({k: v.shape for k, v in tf_weights.items()})
''')

code('''
# --- 3.5 plain Python values and containers --------------------------------
plain = {
    "list": [1, 2, 3, 4, 5],
    "list_mixed": [1, "two", 3.0, True, None],
    "tuple": (1.5, "pi", False),
    "nested": {"a": {"b": {"c": [1, [2, [3]]]}}},
    "set": {3, 1, 2},
    "str_unicode": "Привет, pyferrite! ∑ λ ∞",
    "bytes": bytes(range(8)),
    "int_big": 2 ** 100 + 7,
    "int_neg": -9_007_199_254_740_993,
    "float": 0.1 + 0.2,
    "float_inf": float("inf"),
    "complex": complex(1.5, -2.25),
    "bool": True,
    "none": None,
}
save("plain", plain, ["pkl", "pkl2", "joblib", "pt"])
''')

code('''
# --- 3.6 a numpy element-type zoo -------------------------------------------
zoo = {
    "bool": rng.integers(0, 2, 32).astype(bool),
    "int8": rng.integers(-128, 128, 32).astype(np.int8),
    "uint8": rng.integers(0, 256, 32).astype(np.uint8),
    "int16": rng.integers(-30000, 30000, 32).astype(np.int16),
    "uint16": rng.integers(0, 65000, 32).astype(np.uint16),
    "int32": rng.integers(-2**31, 2**31 - 1, 32).astype(np.int32),
    "uint32": rng.integers(0, 2**32 - 1, 32, dtype=np.uint64).astype(np.uint32),
    "int64": rng.integers(-2**62, 2**62, 32).astype(np.int64),
    "uint64": rng.integers(0, 2**63, 32, dtype=np.uint64),
    "float16": rng.standard_normal(32).astype(np.float16),
    "float32": rng.standard_normal((4, 8)).astype(np.float32),
    "float64": rng.standard_normal((2, 4, 4)),
    "complex64": (rng.standard_normal(16) + 1j * rng.standard_normal(16)).astype(np.complex64),
    "complex128": rng.standard_normal(16) + 1j * rng.standard_normal(16),
    "unicode": np.array(["alpha", "бета", "γάμμα"]),
    "scalar_0d": np.array(3.25),
}
save("zoo", zoo, ["npz", "pkl", "joblib"])
save("zoo_h5", {k: v for k, v in zoo.items() if v.dtype.kind in "biuf"}, ["h5"])
save("zoo_torch", {k: v for k, v in zoo.items()
                   if v.dtype.name in ("bool", "int8", "uint8", "int16", "int32", "int64",
                                       "float16", "float32", "float64", "complex64", "complex128")},
     ["pt"])
for k in ("int32", "float16", "float64", "complex128", "unicode"):
    save(f"zoo_{k}", zoo[k], ["npy"])

# layout and byte-order cases for .npy
m = np.arange(12, dtype=np.float64).reshape(3, 4) * 0.5
np.save(DATA / "layout_fortran.npy", np.asfortranarray(m)); record("layout_fortran.npy", "layout", m)
np.save(DATA / "layout_bigendian.npy", m.astype(">f8"));   record("layout_bigendian.npy", "layout", m)
np.save(DATA / "layout_3d.npy", np.arange(24, dtype=np.int16).reshape(2, 3, 4))
record("layout_3d.npy", "layout", np.arange(24, dtype=np.int16).reshape(2, 3, 4))
''')

code('''
# --- 3.7 a poisoned file: unpickling it in Python would call os.system ------
# Writing it runs nothing; pyferrite must refuse to read it.
import os as _os
class Payload:
    def __reduce__(self):
        return (_os.system, ("echo this-must-never-run",))
pickle.dump({"weights": np.ones(3), "hook": Payload()}, open(DATA / "poisoned.pkl", "wb"), protocol=4)
MANIFEST.append(dict(file="poisoned.pkl", structure="poisoned", must_refuse=True, entries=[]))
''')

code('''
json.dump(MANIFEST, open(DATA / "manifest.json", "w"), ensure_ascii=False, indent=1)
files = sorted(p.name for p in DATA.iterdir() if p.name != "manifest.json")
by_ext = {}
for f in files:
    by_ext[f.rsplit(".", 1)[1]] = by_ext.get(f.rsplit(".", 1)[1], 0) + 1
print(f"{len(files)} files, {sum(len(m['entries']) for m in MANIFEST)} manifest entries")
print(by_ext)
''')

md("## 4. Reference figure (top row of Fig. 3)")

code('''
# Style shared with the Rust figure: the same sizes, colours and limits.
STYLE = {
    "width_in": 7.0, "height_in": 2.6, "dpi": 600, "font": "Times New Roman",
    "font_pt": 10.0, "title_pt": 10.5, "line_pt": 1.6, "marker_pt": 3.6,
    "grid_rgba": [0.5, 0.5, 0.5, 0.35],
    "colors": ["#1f6fb2", "#d1495b", "#2e8b57"],
}
# the exact 256-entry viridis table, so the Rust heatmap uses identical colours
STYLE["viridis"] = [[round(c, 6) for c in plt.get_cmap("viridis")(i / 255)[:3]] for i in range(256)]
STYLE["field_vmax"] = 1.1
json.dump(STYLE, open(FIG / "style.json", "w"), indent=1)

for f in font_manager.findSystemFonts():
    if "Times_New_Roman" in f or "times" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": STYLE["font"], "font.size": STYLE["font_pt"],
    "axes.titlesize": STYLE["title_pt"], "axes.labelsize": STYLE["font_pt"],
    "xtick.labelsize": STYLE["font_pt"], "ytick.labelsize": STYLE["font_pt"],
    "legend.fontsize": STYLE["font_pt"] - 1, "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})
C = STYLE["colors"]; GRID = dict(color=STYLE["grid_rgba"][:3], alpha=STYLE["grid_rgba"][3], lw=0.6)

def reference_figure(tag, series, knn, field):
    fig, ax = plt.subplots(1, 3, figsize=(STYLE["width_in"], STYLE["height_in"]),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.12]))
    fig.subplots_adjust(left=0.075, right=0.945, bottom=0.185, top=0.885, wspace=0.34)
    # (a) time series
    a = ax[0]
    a.plot(series["t"], series["sin"], "-o", color=C[0], lw=STYLE["line_pt"],
           ms=STYLE["marker_pt"], markevery=5, label="sin")
    a.plot(series["t"], series["parabola"], "-s", color=C[1], lw=STYLE["line_pt"],
           ms=STYLE["marker_pt"], markevery=5, label="парабола")
    a.set(xlim=(0, 100), ylim=(-1.3, 1.9), xticks=range(0, 101, 20),
          yticks=[-1, -0.5, 0, 0.5, 1, 1.5], xlabel="t", ylabel="y")
    a.set_title(f"{tag}: временной ряд (.h5)")
    a.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, borderpad=0.3,
             handlelength=1.6, columnspacing=0.8, handletextpad=0.4)
    # (b) k-NN training set
    b = ax[1]
    names = ["setosa", "versicolor", "virginica"]
    for k in range(3):
        sel = knn["y"] == k
        b.scatter(knn["X"][sel, 0], knn["X"][sel, 1], s=STYLE["marker_pt"] ** 2 * 1.6,
                  color=C[k], edgecolor="white", linewidth=0.4, label=names[k], zorder=3)
    b.set(xlim=(-3, 4), ylim=(-3, 5.2), xticks=range(-3, 5), yticks=range(-3, 6),
          xlabel="ГК 1", ylabel="ГК 2")
    b.set_title(f"{tag}: выборка k-NN (.joblib)")
    b.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.95, borderpad=0.3,
             handletextpad=0.1, labelspacing=0.25, markerscale=0.9)
    # (c) field from a torch tensor
    c = ax[2]
    im = c.imshow(field["field"], origin="lower", cmap="viridis", extent=(-3, 3, -3, 3),
                  vmin=0, vmax=STYLE["field_vmax"], interpolation="nearest", aspect="auto")
    c.set(xticks=range(-3, 4), yticks=range(-3, 4), xlabel="x", ylabel="y")
    c.set_title(f"{tag}: тензор PyTorch (.pt)")
    cb = fig.colorbar(im, ax=c, fraction=0.07, pad=0.03, ticks=[0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=STYLE["font_pt"])
    for x in ax[:2]:
        x.grid(True, **GRID); x.set_axisbelow(True)
    c.grid(True, **GRID)
    return fig

fig = reference_figure("Python", series, knn_params, field_obj)
for ext in ("png", "svg"):
    fig.savefig(FIG / f"python_panels.{ext}", dpi=STYLE["dpi"], facecolor="white")
fig.savefig(FIG / "python_panels.tif", dpi=STYLE["dpi"], facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
print("saved", sorted(p.name for p in FIG.iterdir()))
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "validation.ipynb"
nbf.write(nb, out)
print("wrote", out)

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, out)
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    print(o["text"], end="")
    print("executed", out)
