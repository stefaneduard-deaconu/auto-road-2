"""Cost models: the registry, custom callables, the factories, and what is refused.

Every model must return non-negative finite weights, because Dijkstra is silently wrong
otherwise, and must be nameable, because every result row records the cost it came from.
"""
import numpy as np
import pytest

from core import costs
from core.costs import (COST_MODELS, DEFAULT_COST, InvalidCostModel, TIE_BREAK_PER_METER,
                        available, get, gradient_penalised, name_of, register, resolve,
                        validate, weighted_sum)
from core.grid import Grid
from core.search import build_graph, dijkstra

DH = np.array([-12.0, -0.5, 0.0, 0.5, 12.0])
LENGTH = np.array([10.0, 10.0, 10.0, 10.0, 14.142135623730951])


@pytest.fixture(autouse=True)
def _restore_registry():
    """The registry is module state; a test that registers must not leak into the next."""
    original = dict(COST_MODELS)
    yield
    COST_MODELS.clear()
    COST_MODELS.update(original)


# -- the named models --------------------------------------------------------------------

def test_the_four_named_models_are_registered():
    assert available() == ['3d', 'height', 'height_tiebreak', 'length']
    assert DEFAULT_COST in COST_MODELS


@pytest.mark.parametrize('name', ['height', 'height_tiebreak', 'length', '3d'])
def test_every_registered_model_passes_the_probe(name):
    assert validate(name) is get(name)


def test_height_is_equation_1():
    np.testing.assert_allclose(get('height')(DH, LENGTH), np.abs(DH))


def test_height_tiebreak_adds_only_a_tiny_length_term():
    plain, tie = get('height')(DH, LENGTH), get('height_tiebreak')(DH, LENGTH)
    np.testing.assert_allclose(tie - plain, TIE_BREAK_PER_METER * LENGTH)


def test_length_and_3d():
    np.testing.assert_allclose(get('length')(DH, LENGTH), LENGTH)
    np.testing.assert_allclose(get('3d')(DH, LENGTH), np.hypot(DH, LENGTH))


def test_an_unknown_name_lists_what_is_known():
    with pytest.raises(KeyError, match='height_tiebreak'):
        get('nope')


# -- resolving and naming ----------------------------------------------------------------

def test_a_name_and_a_callable_both_resolve():
    assert resolve('height') is get('height')
    mine = lambda dh, length_m: np.abs(dh)          # noqa: E731
    assert resolve(mine) is mine


def test_something_that_is_neither_is_refused():
    with pytest.raises(InvalidCostModel, match='registered name'):
        resolve(42)


def test_a_registered_model_records_its_registry_name():
    """`length_3d.__name__` is 'length_3d' but the registry calls it '3d'."""
    assert name_of('3d') == '3d'
    assert name_of(costs.length_3d) == '3d'


def test_a_plain_function_records_its_own_name():
    def steepness(dh, length_m):
        return np.abs(dh) / length_m
    assert name_of(steepness) == 'steepness'


def test_an_anonymous_model_is_still_nameable():
    assert name_of(lambda dh, length_m: np.abs(dh)).startswith('<unnamed')


# -- the registry ------------------------------------------------------------------------

def test_registering_makes_a_model_selectable_by_name():
    register('uphill_only', lambda dh, length_m: np.clip(dh, 0.0, None))
    assert 'uphill_only' in available()
    assert name_of(get('uphill_only')) == 'uphill_only'
    np.testing.assert_allclose(get('uphill_only')(DH, LENGTH), np.clip(DH, 0.0, None))


def test_rebinding_a_name_is_refused_unless_it_is_deliberate():
    with pytest.raises(InvalidCostModel, match='already registered'):
        register('height', lambda dh, length_m: length_m)
    register('height', lambda dh, length_m: length_m, overwrite=True)
    np.testing.assert_allclose(get('height')(DH, LENGTH), LENGTH)


def test_registering_an_invalid_model_is_refused_at_registration():
    with pytest.raises(InvalidCostModel, match='negative'):
        register('backwards', lambda dh, length_m: -np.abs(dh))
    assert 'backwards' not in COST_MODELS


def test_an_empty_name_is_refused():
    with pytest.raises(InvalidCostModel, match='non-empty'):
        register('', lambda dh, length_m: length_m)


# -- validation --------------------------------------------------------------------------

def test_a_negative_model_is_refused():
    with pytest.raises(InvalidCostModel, match='negative weight'):
        validate(lambda dh, length_m: dh)           # signed, so negative downhill


def test_a_non_finite_model_is_refused():
    with np.errstate(divide='ignore', invalid='ignore'):        # the division is the point
        with pytest.raises(InvalidCostModel, match='non-finite'):
            validate(lambda dh, length_m: np.abs(dh) / 0.0)


def test_a_model_with_the_wrong_output_shape_is_refused():
    with pytest.raises(InvalidCostModel, match='one weight per edge'):
        validate(lambda dh, length_m: np.zeros(3))


def test_a_model_that_raises_is_reported_with_its_exception():
    def broken(dh, length_m):
        raise RuntimeError('boom')
    with pytest.raises(InvalidCostModel, match='RuntimeError'):
        validate(broken)


def test_a_scalar_model_is_accepted_and_broadcast():
    """A constant cost is a legitimate model: it makes Dijkstra count steps."""
    assert validate(lambda dh, length_m: 1.0)


def test_validation_is_a_probe_not_a_proof():
    """Non-negative on the probe, negative on a real terrain: build_graph is the backstop."""
    sneaky = lambda dh, length_m: dh + 12.0         # noqa: E731
    validate(sneaky)                                 # the probe's dh never goes below -12
    grid = Grid(surf=np.array([[0.0, 100.0], [0.0, 0.0]]), cell_size_m=10.0)
    with pytest.raises(ValueError, match='negative weight'):
        build_graph(grid, cost=sneaky)


# -- weighted_sum ------------------------------------------------------------------------

def test_weighted_sum_reproduces_the_named_models():
    np.testing.assert_allclose(weighted_sum(height=1.0)(DH, LENGTH), get('height')(DH, LENGTH))
    np.testing.assert_allclose(weighted_sum(length=1.0)(DH, LENGTH), get('length')(DH, LENGTH))
    np.testing.assert_allclose(weighted_sum(length_3d=1.0)(DH, LENGTH), get('3d')(DH, LENGTH))
    np.testing.assert_allclose(
        weighted_sum(height=1.0, length=TIE_BREAK_PER_METER)(DH, LENGTH),
        get('height_tiebreak')(DH, LENGTH))


def test_weighted_sum_blends():
    model = weighted_sum(height=2.0, length=0.5)
    np.testing.assert_allclose(model(DH, LENGTH), 2.0 * np.abs(DH) + 0.5 * LENGTH)


def test_weighted_sum_carries_its_parameters_in_its_name():
    name = name_of(weighted_sum(height=1.0, length=0.5))
    assert 'weighted_sum' in name and 'height=1.0' in name and 'length=0.5' in name


def test_a_negative_weight_is_refused_at_construction():
    with pytest.raises(InvalidCostModel, match='non-negative'):
        weighted_sum(height=-1.0)


def test_an_all_zero_blend_is_refused():
    with pytest.raises(InvalidCostModel, match='non-zero'):
        weighted_sum()


# -- gradient_penalised ------------------------------------------------------------------

def test_gradient_penalised_leaves_gentle_edges_alone():
    model = gradient_penalised('height', i_max_percent=7.0)
    gentle_dh, gentle_len = np.array([0.5]), np.array([10.0])     # 5%
    np.testing.assert_allclose(model(gentle_dh, gentle_len), np.abs(gentle_dh))


def test_gradient_penalised_charges_per_metre_per_percent_over():
    model = gradient_penalised('height', i_max_percent=7.0, penalty_per_percent=2.0)
    dh, length_m = np.array([1.2]), np.array([10.0])              # 12%, 5 over
    np.testing.assert_allclose(model(dh, length_m), 1.2 + 2.0 * 5.0 * 10.0)


def test_gradient_penalised_is_scale_free_in_the_cells():
    """The same slope cut into 1 m or 10 m cells costs the same per metre of road."""
    model = gradient_penalised('length', i_max_percent=7.0)
    fine = model(np.array([0.12]), np.array([1.0]))
    coarse = model(np.array([1.2]), np.array([10.0]))
    assert coarse[0] / fine[0] == pytest.approx(10.0)


def test_gradient_penalised_accepts_a_custom_base():
    model = gradient_penalised(lambda dh, length_m: np.zeros_like(dh), i_max_percent=7.0,
                               penalty_per_percent=1.0)
    np.testing.assert_allclose(model(np.array([1.2]), np.array([10.0])), 5.0 * 10.0)


def test_gradient_penalised_names_its_base_and_parameters():
    name = name_of(gradient_penalised('height', i_max_percent=7.0))
    assert 'gradient_penalised(height' in name and 'i_max_percent=7.0' in name


@pytest.mark.parametrize('kwargs', [{'i_max_percent': 0.0}, {'i_max_percent': -1.0},
                                    {'i_max_percent': 7.0, 'penalty_per_percent': -1.0}])
def test_gradient_penalised_refuses_nonsense_parameters(kwargs):
    with pytest.raises(InvalidCostModel):
        gradient_penalised('height', **kwargs)


# -- the models actually drive the search ------------------------------------------------

def staircase() -> Grid:
    """Flat along the columns, a 20% step down the rows: the two objectives disagree."""
    surf = np.tile(np.arange(12, dtype=float)[:, None] * 2.0, (1, 12))
    return Grid(surf=surf, cell_size_m=10.0)


@pytest.mark.parametrize('cost', ['height', 'height_tiebreak', 'length', '3d'])
def test_every_named_model_produces_a_searchable_graph(cost):
    grid = staircase()
    result = dijkstra(build_graph(grid, cost=cost), grid.index((0, 0)), grid.index((11, 11)))
    assert result.reached
    assert result.cost >= 0


def test_a_custom_callable_can_be_searched_without_registering_it():
    grid = staircase()
    graph = build_graph(grid, cost=lambda dh, length_m: np.abs(dh) + length_m)
    result = dijkstra(graph, grid.index((0, 0)), grid.index((11, 11)))
    assert result.reached
    assert graph.cost_name.startswith('<unnamed')


def test_the_graph_records_the_name_of_whatever_it_was_built_with():
    grid = staircase()
    assert build_graph(grid, cost='length').cost_name == 'length'
    model = weighted_sum(height=1.0, length=0.25)
    assert build_graph(grid, cost=model).cost_name == name_of(model)


def test_a_constant_cost_makes_dijkstra_count_steps():
    grid = staircase()
    graph = build_graph(grid, cost=lambda dh, length_m: 1.0)
    result = dijkstra(graph, grid.index((0, 0)), grid.index((11, 11)))
    #: 8-connected, so the diagonal gets there in 11 steps
    assert result.cost == pytest.approx(11.0)


def test_the_cost_model_changes_which_path_is_chosen():
    """`length` cuts straight down the slope; a gradient penalty makes it traverse."""
    grid = staircase()
    start, target = grid.index((0, 0)), grid.index((11, 11))
    direct = dijkstra(build_graph(grid, cost='length'), start, target)
    penalised = dijkstra(build_graph(grid, cost=gradient_penalised(
        'length', i_max_percent=7.0, penalty_per_percent=10.0)), start, target)
    assert direct.reached and penalised.reached
    #: the penalised path is longer in metres, which is the point of paying for gradient
    assert len(penalised.path) >= len(direct.path)


def test_the_gradient_penalty_reduces_the_worst_gradient_on_a_real_choice():
    """A cheap gentle detour beside a steep direct line: the penalty should take the detour."""
    surf = np.zeros((9, 9))
    surf[:, :] = np.arange(9)[None, :] * 0.2          # a gentle 2% ramp across the columns
    surf[4, :] = np.arange(9) * 3.0                   # one steep row through the middle
    grid = Grid(surf=surf, cell_size_m=10.0)
    start, target = grid.index((4, 0)), grid.index((4, 8))

    from core.metrics import i_max_percent
    plain = dijkstra(build_graph(grid, cost='height_tiebreak'), start, target)
    penalised = dijkstra(build_graph(grid, cost=gradient_penalised(
        'height_tiebreak', i_max_percent=7.0, penalty_per_percent=100.0)), start, target)
    worst = {}
    for label, result in (('plain', plain), ('penalised', penalised)):
        xyz = grid.path_xyz_m(result.coords(grid).astype(float))
        worst[label] = i_max_percent(xyz)
    assert worst['penalised'] <= worst['plain']
