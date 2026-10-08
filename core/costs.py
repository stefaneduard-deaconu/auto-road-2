"""Edge cost models: what the search minimises.

A cost model turns the `(dh, length_m)` arrays of every directed edge into a non-negative
weight array. Anything callable with that signature is a model, so a caller may pass one of
the registered names, a function of their own, or one built by a factory here.

`height` is the objective of Equation 1, the sum of absolute height
differences along the path. It ignores horizontal distance and is completely flat on a
plateau, so the shortest path is massively degenerate there and the "spatial difference vs
the full-grid path" becomes noise (bad point 9 / S2 item 12). `height_tiebreak` adds a tiny
length term that breaks those ties deterministically without changing which paths are
optimal for the height objective, as long as the terrain's height differences are far larger
than `TIE_BREAK_PER_METER * total length`.

**Every model must return non-negative, finite weights.** Dijkstra is only correct for
non-negative edge weights, and `validate` checks it on a probe before a wrong answer can be
produced quietly. `build_graph` re-checks the real weights, because a model can be
non-negative on the probe and negative on a real terrain.

**A model has to be nameable.** Every result row records the cost it was produced with, so `name_of` gives a stable string for a registered name, a
plain function, or a factory-built closure whose parameters are part of its name:
`weighted_sum(height=1.0,length=0.5,length_3d=0.0)` identifies itself completely.
"""
from __future__ import annotations

from typing import Callable, Final, Optional, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: the array type every model takes and returns
FloatArray: TypeAlias = NDArray[np.float64]


@runtime_checkable
class CostModel(Protocol):
    """`(dh, length_m) -> weight`, elementwise over the edges of a grid graph.

    `dh` is `height(v) - height(u)` in metres and is SIGNED; `length_m` is the planar step
    length and is always positive. The result must be non-negative and finite, and must
    broadcast to the shape of the inputs (a scalar is accepted, so a constant-cost model is
    one line).
    """

    def __call__(self, dh: FloatArray, length_m: FloatArray) -> FloatArray: ...


#: a registered name or a model itself: what every `cost=` parameter accepts
CostSpec: TypeAlias = str | CostModel

#: cost added per metre of horizontal length in the `height_tiebreak` model. Small enough
#: that it cannot reorder two paths whose height cost differs by more than
#: TIE_BREAK_PER_METER * (length difference); on a 1 km terrain that is 1e-3 m of height.
TIE_BREAK_PER_METER: Final[float] = 1e-6

#: default strength of `gradient_penalised`: one length unit of extra cost per metre
#: travelled per percent over the limit
DEFAULT_GRADIENT_PENALTY: Final[float] = 1.0


class InvalidCostModel(ValueError):
    """A cost model that Dijkstra cannot use, or one whose output does not fit the edges."""


# -- the named models ------------------------------------------------------------------

def height(dh: FloatArray, length_m: FloatArray) -> FloatArray:
    """Equation 1: sum of |height difference|."""
    return np.abs(dh)


def height_tiebreak(dh: FloatArray, length_m: FloatArray) -> FloatArray:
    """Equation 1 with a deterministic tie-break on horizontal length."""
    return np.abs(dh) + TIE_BREAK_PER_METER * length_m


def length(dh: FloatArray, length_m: FloatArray) -> FloatArray:
    """Planar length in metres."""
    return np.asarray(length_m, dtype=float)


def length_3d(dh: FloatArray, length_m: FloatArray) -> FloatArray:
    """3D length in metres."""
    return np.hypot(np.asarray(dh, dtype=float), np.asarray(length_m, dtype=float))


# -- factories: models with parameters ---------------------------------------------------

def named(model: Callable[..., FloatArray], name: str) -> CostModel:
    """Attach the name a result row will record. Returns the same object."""
    model.cost_name = name          # type: ignore[attr-defined]
    return model                    # type: ignore[return-value]


def weighted_sum(*, height: float = 0.0, length: float = 0.0,
                 length_3d: float = 0.0) -> CostModel:
    """A linear blend of the three basic terms, e.g. "metres climbed plus half the metres run".

    Every named model above is a special case: `height=1` is Equation 1, `length=1` is the
    planar shortest path, and `height=1, length=TIE_BREAK_PER_METER` is `height_tiebreak`.
    All three weights must be non-negative, or the sum could go negative and Dijkstra would
    be wrong.
    """
    weights = {'height': height, 'length': length, 'length_3d': length_3d}
    negative = sorted(key for key, value in weights.items() if value < 0)
    if negative:
        raise InvalidCostModel(f"weights must be non-negative; {negative} are not")
    if not any(weights.values()):
        raise InvalidCostModel("at least one weight must be non-zero, or every path is free")

    def model(dh: FloatArray, length_m: FloatArray) -> FloatArray:
        out = np.zeros(np.broadcast(dh, length_m).shape, dtype=float)
        if height:
            out += height * np.abs(dh)
        if length:
            out += length * length_m
        if length_3d:
            out += length_3d * np.hypot(dh, length_m)
        return out

    model.native_spec = ('weighted', float(height), float(length), float(length_3d))
    return named(model, f"weighted_sum(height={height!r},length={length!r},"
                        f"length_3d={length_3d!r})")


def gradient_penalised(base: CostSpec = 'height_tiebreak', *, i_max_percent: float,
                       penalty_per_percent: float = DEFAULT_GRADIENT_PENALTY) -> CostModel:
    """`base`, plus a penalty per metre for every percent by which an edge exceeds `i_max`.

    The soft counterpart of the hard edge cut in `core.od_feasibility`: that one deletes
    steep edges and so can disconnect the graph, while this one keeps every edge and merely
    makes steep ones expensive, so a path always exists. It exists because the search
    otherwise never sees the road class at all and will walk past a
    gentle alignment that holds `i_max` in favour of a steep one that does not.

    The penalty is proportional to length, so it is a cost per metre of over-steep road and
    does not depend on how the same slope happens to be cut into cells.
    """
    if i_max_percent <= 0:
        raise InvalidCostModel(f"i_max_percent must be positive; got {i_max_percent}")
    if penalty_per_percent < 0:
        raise InvalidCostModel(
            f"penalty_per_percent must be non-negative; got {penalty_per_percent}")
    base_model = resolve(base)

    def model(dh: FloatArray, length_m: FloatArray) -> FloatArray:
        gradient = 100.0 * np.abs(dh) / length_m
        excess = np.clip(gradient - i_max_percent, 0.0, None)
        return np.asarray(base_model(dh, length_m), dtype=float) \
            + penalty_per_percent * excess * length_m

    model.native_spec = ('penalty', native_spec(base), float(i_max_percent),
                         float(penalty_per_percent))
    return named(model, f"gradient_penalised({name_of(base)},i_max_percent={i_max_percent!r},"
                        f"penalty_per_percent={penalty_per_percent!r})")


# -- the registry ------------------------------------------------------------------------

COST_MODELS: dict[str, CostModel] = {
    'height': height,
    'height_tiebreak': height_tiebreak,
    'length': length,
    '3d': length_3d,
}

#: so that `name_of(length_3d)` records '3d', the name the registry knows it by, rather
#: than the function's own `__name__`
for _registered_name, _registered_model in COST_MODELS.items():
    named(_registered_model, _registered_name)
del _registered_name, _registered_model

#: what the experiments use unless told otherwise
DEFAULT_COST: Final[str] = 'height_tiebreak'

#: how `core.native` reproduces each registered model; a callable without a spec is
#: Python-only.
_NATIVE_SPECS: dict[str, tuple] = {
    'height': ('height',),
    'height_tiebreak': ('height_tiebreak', TIE_BREAK_PER_METER),
    'length': ('length',),
    '3d': ('3d',),
}


def native_spec(spec: CostSpec) -> Optional[tuple]:
    """The compiled engine's description of a cost model, or None if it has none."""
    if isinstance(spec, str):
        return _NATIVE_SPECS.get(spec)
    registered = getattr(spec, 'cost_name', None)
    if registered in _NATIVE_SPECS and COST_MODELS.get(registered) is spec:
        return _NATIVE_SPECS[registered]
    return getattr(spec, 'native_spec', None)


def available() -> list[str]:
    """The registered names, sorted."""
    return sorted(COST_MODELS)


def register(name: str, model: CostModel, *, overwrite: bool = False) -> CostModel:
    """Add a model under `name` so it can be selected by string, e.g. from a CLI.

    Refuses to replace an existing name unless `overwrite=True`: a result row records only
    the name, so silently rebinding one would make two different runs indistinguishable.
    """
    if not name:
        raise InvalidCostModel('a cost model needs a non-empty name')
    if name in COST_MODELS and not overwrite:
        raise InvalidCostModel(
            f"{name!r} is already registered; pass overwrite=True if that is deliberate. "
            f"Rebinding a name makes two runs that recorded it indistinguishable.")
    validate(model, name=name)
    COST_MODELS[name] = named(model, name)
    return model


def get(name: str) -> CostModel:
    """The registered model called `name`."""
    try:
        return COST_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown cost model {name!r}; known: {available()}") from None


def resolve(spec: CostSpec) -> CostModel:
    """A name or a callable, either way a model. This is what `cost=` parameters accept."""
    if isinstance(spec, str):
        return get(spec)
    if callable(spec):
        return spec
    raise InvalidCostModel(
        f"a cost must be a registered name ({available()}) or a callable "
        f"(dh, length_m) -> weights; got {type(spec).__name__}")


def name_of(spec: CostSpec) -> str:
    """The string a result row records for `spec`.

    A registered name is itself; a factory-built model carries its parameters in
    `cost_name`; a plain function falls back to its `__name__`.
    """
    if isinstance(spec, str):
        return spec
    name = getattr(spec, 'cost_name', None)
    if isinstance(name, str) and name:
        return name
    name = getattr(spec, '__name__', None)
    if isinstance(name, str) and name and name != '<lambda>':
        return name
    return f"<unnamed {type(spec).__name__}>"


#: signed height differences and the matching step lengths that `validate` probes with:
#: downhill, flat, uphill, and a diagonal step, all with a positive length
PROBE_DH: Final[FloatArray] = np.array([-12.0, -0.5, 0.0, 0.5, 12.0, 3.0])
PROBE_LENGTH_M: Final[FloatArray] = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 14.142135623730951])


def validate(spec: CostSpec, *, name: str | None = None) -> CostModel:
    """Run `spec` on a probe and refuse anything Dijkstra cannot use. Returns the model.

    Checks that it is callable with `(dh, length_m)`, that the result broadcasts to the edge
    shape, and that it is finite and non-negative. It is a probe, not a proof: a model can
    pass here and still produce a negative weight on a real terrain, which is why
    `core.search.build_graph` checks the real weights too.
    """
    model = resolve(spec)
    label = name or name_of(spec)
    try:
        raw = model(PROBE_DH, PROBE_LENGTH_M)
    except Exception as error:
        raise InvalidCostModel(
            f"cost model {label!r} raised {type(error).__name__} on the probe edges: "
            f"{error}. It must accept (dh, length_m) as float arrays.") from error

    out = np.asarray(raw, dtype=float)
    try:
        out = np.broadcast_to(out, PROBE_DH.shape)
    except ValueError:
        raise InvalidCostModel(
            f"cost model {label!r} returned shape {out.shape} for {PROBE_DH.shape} edges; "
            f"it must return one weight per edge, or a scalar.") from None
    if not np.isfinite(out).all():
        raise InvalidCostModel(f"cost model {label!r} returned a non-finite weight: {out}")
    if (out < 0).any():
        raise InvalidCostModel(
            f"cost model {label!r} returned a negative weight ({out.min()}); Dijkstra is "
            f"only correct for non-negative weights.")
    return model
