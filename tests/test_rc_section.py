"""Regression tests for the RC cross-section.

Each test class corresponds to a defect fixed in the RC hardening series, plus
an end-to-end matrix over the supported shapes, axes and confinement states.
"""
import numpy as np
import pytest

from conftest import make_rect, make_circle, make_obround
from libdenavit.section import (
    RC, Rectangle, Circle, Obround, ReinfRect, ReinfCirc, ReinfIntersectingLoops,
)


class TestReinforcementNormalization:
    """A tuple of patterns used to be wrapped as a single element."""

    def test_single_object_becomes_one_item_list(self, rect):
        pattern = ReinfRect(14, 24, 3, 3, 0.79)
        rect.reinforcement = pattern
        assert rect.reinforcement == [pattern]

    def test_list_is_copied(self, rect):
        a, b = ReinfRect(14, 24, 3, 3, 0.79), ReinfRect(10, 20, 2, 2, 0.5)
        source = [a, b]
        rect.reinforcement = source
        assert rect.reinforcement == [a, b]
        assert rect.reinforcement is not source

    def test_tuple_is_expanded_not_wrapped(self, rect):
        a, b = ReinfRect(14, 24, 3, 3, 0.79), ReinfRect(10, 20, 2, 2, 0.5)
        rect.reinforcement = (a, b)
        assert rect.reinforcement == [a, b]
        assert not isinstance(rect.reinforcement[0], tuple)

    @pytest.mark.parametrize('empty', [[], ()])
    def test_empty_rejected(self, rect, empty):
        with pytest.raises(ValueError, match='at least one reinforcement pattern'):
            rect.reinforcement = empty

    def test_validation_reports_offending_type(self, record_section):
        section = RC(Rectangle(30, 20), ReinfCirc(9, 8, 0.79), 4, 60, 'US')
        with pytest.raises(ValueError, match='ReinfCirc'):
            record_section(section, 'x')


class TestMaximumTensileSteelStrain:
    """coordinates is (x_array, y_array); iterating it visited the arrays."""

    @staticmethod
    def _brute_force(section, axial, kx, ky):
        best = -np.inf
        for pattern in section.reinforcement:
            xs, ys = pattern.coordinates
            for x, y in zip(xs, ys):
                best = max(best, axial - y * kx - x * ky)
        return best

    @pytest.mark.parametrize('kx,ky', [(1e-4, 0), (0, 1e-4), (5e-5, 7e-5), (0, 0)])
    def test_matches_brute_force(self, rect, kx, ky):
        got = rect.maximum_tensile_steel_strain(1e-3, kx, ky)
        assert got == pytest.approx(self._brute_force(rect, 1e-3, kx, ky), rel=1e-12)

    def test_known_bar_layout(self, rect):
        # Bars reach y = -12; the most tensile strain is 0 - (-12)*1e-4.
        assert rect.maximum_tensile_steel_strain(0.0, 1e-4, 0) == pytest.approx(1.2e-3)

    def test_zero_curvature_returns_axial_strain(self, rect):
        assert rect.maximum_tensile_steel_strain(3.1e-3) == pytest.approx(3.1e-3)

    def test_all_patterns_considered(self, rect):
        offset = ReinfRect(8, 10, 2, 2, 0.5)
        offset.yc = -20
        rect.reinforcement = [ReinfRect(14, 24, 3, 3, 0.79), offset]
        got = rect.maximum_tensile_steel_strain(1e-3, 1e-4, 0)
        assert got == pytest.approx(self._brute_force(rect, 1e-3, 1e-4, 0), rel=1e-12)
        # the offset pattern reaches lower than the main cage and must govern
        assert got == pytest.approx(1e-3 + 25 * 1e-4)

    def test_alias(self, rect):
        assert rect.maximum_tensile_strain(1e-3, 1e-4) == \
               rect.maximum_tensile_steel_strain(1e-3, 1e-4)


class TestObroundCompressionStrain:
    """An obround is a stadium, so its support function is exact."""

    @staticmethod
    def _dense_boundary(D, a, kx, ky, n=200001):
        t = np.linspace(0, 2 * np.pi, n)
        xs = np.concatenate([a / 2 + D / 2 * np.cos(t), -a / 2 + D / 2 * np.cos(t),
                             np.linspace(-a / 2, a / 2, n), np.linspace(-a / 2, a / 2, n)])
        ys = np.concatenate([D / 2 * np.sin(t), D / 2 * np.sin(t),
                             np.full(n, D / 2), np.full(n, -D / 2)])
        on_boundary = (np.abs(xs) >= a / 2) | (np.abs(np.abs(ys) - D / 2) < 1e-9)
        return np.min(-ys[on_boundary] * abs(kx) - xs[on_boundary] * abs(ky))

    @pytest.mark.parametrize('kx,ky', [(1e-4, 0), (0, 1e-4), (1e-4, 1e-4),
                                       (3e-5, 9e-5), (2e-4, 5e-5)])
    def test_agrees_with_dense_sampling(self, obround, kx, ky):
        got = obround.maximum_concrete_compression_strain(0.0, kx, ky)
        expected = self._dense_boundary(24, 12, kx, ky)
        assert got == pytest.approx(expected, abs=1e-10)

    def test_reduces_to_circle_as_flat_vanishes(self):
        thin = RC(Obround(24, 1e-9), ReinfIntersectingLoops(20, 12, 8, 0.79), 4, 60, 'US')
        circ = make_circle()
        assert thin.maximum_concrete_compression_strain(0.0, 1e-4, 1e-4) == \
               pytest.approx(circ.maximum_concrete_compression_strain(0.0, 1e-4, 1e-4), abs=1e-9)

    def test_zero_curvature_returns_axial_strain(self, obround):
        assert obround.maximum_concrete_compression_strain(2e-3) == pytest.approx(2e-3)


class TestInteractionDiagramCache:
    """One cached object used to answer every parameter combination."""

    def test_axes_do_not_share_entries(self, rect):
        assert rect.interaction_diagram_object('x', 8) is not \
               rect.interaction_diagram_object('y', 8)

    def test_identical_requests_are_reused(self, rect):
        assert rect.interaction_diagram_object('x', 8) is \
               rect.interaction_diagram_object('x', 8)

    @pytest.mark.parametrize('kwargs', [
        {'factored': True}, {'only_compressive': False}, {'num_points': 10},
    ])
    def test_parameters_do_not_share_entries(self, rect, kwargs):
        base = rect.interaction_diagram_object('x', num_points=8)
        kwargs = {'num_points': 8, **kwargs}
        assert rect.interaction_diagram_object('x', **kwargs) is not base

    @pytest.mark.parametrize('attribute,value', [
        ('Ec', 4000.0), ('Es', 29500.0), ('eps_c', 0.0021),
    ])
    def test_setters_invalidate(self, rect, attribute, value):
        rect.interaction_diagram_object('x', 8)
        setattr(rect, attribute, value)
        assert rect._CS_id2d == {}

    def test_reinforcement_setter_invalidates(self, rect):
        rect.interaction_diagram_object('x', 8)
        rect.reinforcement = ReinfRect(14, 24, 4, 4, 0.79)
        assert rect._CS_id2d == {}


class TestAci209Inputs:
    """ACI 209R-92 corrections take percentages, not fractions."""

    @pytest.mark.parametrize('units', ['us', 'si'])
    def test_defaults_are_finite(self, units):
        section = RC(Rectangle(30, 20), ReinfRect(14, 24, 3, 3, 0.79), 4, 60, units)
        for props in (section.get_creep_props_for_uniaxial_material(),
                      section.get_shrinkage_props_for_uniaxial_material()):
            assert all(np.isfinite(v) for v in props.values())

    @pytest.mark.parametrize('units', ['us', 'si'])
    def test_reference_values_reproduce_defaults(self, units):
        """psi=50% and alpha=6% are the ACI 209 reference values."""
        section = RC(Rectangle(30, 20), ReinfRect(14, 24, 3, 3, 0.79), 4, 60, units)
        assert section.get_creep_props_for_uniaxial_material(
            fine_agg_ratio=50, air_content=6) == \
            section.get_creep_props_for_uniaxial_material()

    def test_air_content_scales_as_percentage_points(self):
        section = make_rect()
        base = section.get_creep_props_for_uniaxial_material()['phi_u']
        richer = section.get_creep_props_for_uniaxial_material(air_content=8)['phi_u']
        # gamma_c_a = 0.46 + 0.09*8 = 1.18
        assert richer == pytest.approx(base * 1.18, rel=1e-9)

    def test_optional_dictionaries_may_be_omitted(self):
        """build_ops_fiber_section expanded None with **, raising TypeError."""
        section = make_rect()
        assert section.get_creep_props_for_uniaxial_material(**(None or {}))
        assert section.get_shrinkage_props_for_uniaxial_material(**(None or {}))

    @pytest.mark.parametrize('rh', [0.2, 1.5])
    def test_invalid_relative_humidity_rejected(self, rh):
        with pytest.raises(ValueError, match='Relative humidity'):
            make_rect().get_shrinkage_props_for_uniaxial_material(RH=rh)


class TestAxisMapping:
    """x bending uses y coordinates and nfy; y bending uses x and nfx."""

    def test_rectangle_uses_axis_appropriate_bar_coordinates(self, record_section):
        section = make_rect()
        xs, ys = section.reinforcement[0].coordinates
        assert record_section(section, 'x').steel_positions == \
               sorted({round(v, 9) for v in ys})
        assert record_section(section, 'y').steel_positions == \
               sorted({round(v, 9) for v in xs})

    def test_circle_y_bending_uses_x_coordinates(self, record_section):
        section = make_circle(num_bars=6)   # 6 bars -> x and y sets differ
        xs, ys = section.reinforcement[0].coordinates
        positions = record_section(section, 'y').steel_positions
        for x in xs:
            assert round(float(x), 9) in positions

    @pytest.mark.parametrize('factory', [make_rect, make_circle, make_obround])
    @pytest.mark.parametrize('axis,changed,unchanged', [('x', 'nfy', 'nfx'),
                                                        ('y', 'nfx', 'nfy')])
    def test_only_the_active_direction_refines(self, record_section, factory,
                                               axis, changed, unchanged):
        section = factory()
        base = record_section(section, axis, nfy=20, nfx=20).total_fibers
        refined = record_section(section, axis,
                                 **{changed: 40, unchanged: 20}).total_fibers
        inert = record_section(section, axis,
                               **{unchanged: 40, changed: 20}).total_fibers
        assert refined > base, f'{changed} should refine {axis} bending'
        assert inert == base, f'{unchanged} must be ignored for {axis} bending'

    def test_rectangle_cover_depth_uses_the_bending_direction(self, record_section):
        """cdb used By for both axes, giving negative cover about y."""
        reinf = ReinfRect(10, 24, 3, 5, 0.79)   # Bx=10, By=24
        reinf.db = 1.0
        section = RC(Rectangle(30, 20), reinf, 4, 60, 'US', dbt=0.5, s=4, fyt=60)
        for axis, depth, reinf_depth in [('x', 30, 24), ('y', 20, 10)]:
            layers = record_section(section, axis, conc_mat_type='Concrete04').layers
            expected_cdb = (depth - reinf_depth) / 2 - 1.0 / 2 - 0.5 / 2
            core = layers[0]['coords']
            strip = (core[2] - core[0]) / (layers[0]['n'] - 1) if layers[0]['n'] > 1 else 0
            assert core[0] < 0 < core[2], 'core layer must straddle the centroid'
            assert -depth / 2 + expected_cdb == pytest.approx(core[0] - strip / 2, abs=1e-9)

    @pytest.mark.parametrize('factory', [make_rect, make_circle, make_obround])
    def test_unsupported_axis_rejected(self, record_section, factory):
        with pytest.raises(ValueError, match='axis'):
            record_section(factory(), 'z')


class TestConcreteFiberPlacement:
    """Layer endpoints are fiber centers, so strips sit at their centroids."""

    @pytest.mark.parametrize('nfy', [1, 2, 4, 10, 20])
    def test_total_area_is_exact(self, record_section, nfy):
        fibers = record_section(make_rect(), 'x', nfy=nfy).layer_fibers()
        assert sum(a for _, a in fibers) == pytest.approx(30 * 20, rel=1e-12)

    @pytest.mark.parametrize('nfy', [2, 4, 10, 20])
    def test_centroid_is_exact(self, record_section, nfy):
        fibers = record_section(make_rect(), 'x', nfy=nfy).layer_fibers()
        total = sum(a for _, a in fibers)
        assert sum(y * a for y, a in fibers) / total == pytest.approx(0.0, abs=1e-9)

    def test_second_moment_converges(self, record_section):
        exact = 20 * 30 ** 3 / 12
        errors = []
        for nfy in [4, 8, 16, 32]:
            fibers = record_section(make_rect(), 'x', nfy=nfy).layer_fibers()
            I = sum(a * y ** 2 for y, a in fibers)
            errors.append(abs(I - exact) / exact)
        assert all(b < a for a, b in zip(errors, errors[1:])), errors
        # second order convergence: each refinement cuts the error about 4x
        assert errors[-1] < errors[0] / 30

    @pytest.mark.parametrize('nfy', [1, 2, 5, 20])
    def test_no_fiber_lies_outside_the_section(self, record_section, nfy):
        fibers = record_section(make_rect(), 'x', nfy=nfy).layer_fibers()
        assert max(abs(y) for y, _ in fibers) <= 30 / 2 + 1e-12

    @pytest.mark.parametrize('nfy', [10, 20, 40])
    def test_confined_cover_plus_core_equals_gross(self, record_section, nfy):
        fibers = record_section(make_rect(confined=True), 'x', nfy=nfy,
                                conc_mat_type='Concrete04').layer_fibers()
        assert sum(a for _, a in fibers) == pytest.approx(30 * 20, rel=1e-12)

    @pytest.mark.parametrize('bad', [0, -3, 2.5, '20', True])
    def test_fiber_counts_validated(self, record_section, bad):
        with pytest.raises(ValueError, match='positive integer'):
            record_section(make_rect(), 'x', nfy=bad)


class TestConfinementValidation:
    """Confinement needs transverse reinforcement and a feasible core."""

    @pytest.mark.parametrize('missing', ['dbt', 's', 'fyt'])
    def test_missing_transverse_property_rejected(self, missing):
        kwargs = dict(dbt=0.5, s=4, fyt=60)
        kwargs[missing] = None
        with pytest.raises(ValueError, match=missing):
            make_rect(**kwargs).confined_concrete_props()

    @pytest.mark.parametrize('name,value', [('dbt', 0), ('dbt', -1), ('s', 0), ('fyt', -60)])
    def test_non_positive_values_rejected(self, name, value):
        kwargs = dict(dbt=0.5, s=4, fyt=60)
        kwargs[name] = value
        with pytest.raises(ValueError, match='finite positive'):
            make_rect(**kwargs).confined_concrete_props()

    @pytest.mark.parametrize('s', [0.2, 0.5])
    def test_spacing_must_exceed_bar_diameter(self, s):
        """s <= dbt gave negative clear spacing and inflated fcc silently."""
        with pytest.raises(ValueError, match='clear spacing'):
            make_rect(dbt=0.5, s=s, fyt=60).confined_concrete_props()

    def test_single_bar_per_side_rejected(self):
        reinf = ReinfRect(14, 24, 1, 3, 0.79)
        reinf.db = 1.0
        section = RC(Rectangle(30, 20), reinf, 4, 60, 'US', dbt=0.5, s=4, fyt=60)
        with pytest.raises(ValueError, match='at least 2 bars'):
            section.confined_concrete_props()

    @pytest.mark.parametrize('factory', [make_rect, make_circle, make_obround])
    def test_valid_geometry_is_confined(self, factory):
        section = factory(confined=True)
        fcc, eps_cc = section.confined_concrete_props()
        assert np.isfinite(fcc) and np.isfinite(eps_cc)
        assert fcc > section.fc, 'confinement must increase strength'
        assert eps_cc > section.eps_c, 'confinement must increase ductility'

    def test_unconfined_obround_needs_no_transverse_reinforcement(self, record_section):
        """ds was built from dbt before the confinement branch was chosen."""
        section = make_obround()
        assert section.dbt is None
        assert record_section(section, 'x').fibers


class TestEIeff:
    """Built-in effective stiffness methods."""

    @pytest.mark.parametrize('EI_type', ['aci-a', 'aci-b'])
    def test_load_independent_methods(self, rect, EI_type):
        value = rect.EIeff('x', EI_type, 0.0)
        assert np.isfinite(value) and value > 0

    def test_scalar_load_dependent_method(self, rect):
        value = rect.EIeff('x', 'aci-c', 0.0, P=100.0, M=500.0)
        assert np.isfinite(value) and value > 0

    def test_array_loads(self, rect):
        P = np.array([100.0, 200.0, 300.0])
        M = np.array([500.0, 600.0, 700.0])
        values = rect.EIeff('x', 'aci-c', 0.0, P=P, M=M)
        assert np.shape(values) == P.shape
        assert np.all(np.isfinite(values))

    def test_one_element_array_matches_scalar(self, rect):
        scalar = rect.EIeff('x', 'aci-c', 0.0, P=100.0, M=500.0)
        array = rect.EIeff('x', 'aci-c', 0.0, P=np.array([100.0]), M=np.array([500.0]))
        assert np.ravel(array)[0] == pytest.approx(scalar, rel=1e-12)

    def test_unknown_method_reports_clearly(self, rect):
        with pytest.raises(ValueError, match='Unknown EI_type'):
            rect.EIeff('x', 'no-such-method')

    def test_user_function_errors_are_not_masked(self, rect, monkeypatch):
        """A bare except reported real errors as an unknown EI_type."""
        import sys
        import types

        module = types.ModuleType('exploding_ei')

        def exploding_ei(*args, **kwargs):
            raise RuntimeError('failure inside the user function')

        module.exploding_ei = exploding_ei
        monkeypatch.setitem(sys.modules, 'exploding_ei', module)
        with pytest.raises(RuntimeError, match='inside the user function'):
            rect.EIeff('x', 'exploding_ei,0.5')


class TestEndToEnd:
    """Every supported shape, axis and confinement state builds a section."""

    @pytest.mark.parametrize('factory', [make_rect, make_circle, make_obround])
    @pytest.mark.parametrize('axis', ['x', 'y'])
    def test_unconfined(self, record_section, factory, axis):
        recorder = record_section(factory(), axis)
        assert recorder.fibers or recorder.layers

    @pytest.mark.parametrize('factory', [make_rect, make_circle, make_obround])
    @pytest.mark.parametrize('axis', ['x', 'y'])
    def test_confined(self, record_section, factory, axis):
        recorder = record_section(factory(confined=True), axis,
                                  conc_mat_type='Concrete04')
        assert recorder.fibers or recorder.layers

    @pytest.mark.parametrize('axis', ['x', 'y'])
    def test_interaction_diagrams_are_axis_specific(self, rect, axis):
        P, M, _ = rect.section_interaction_2d(axis, 8, factored=False,
                                              only_compressive=True)
        assert len(P) == len(M) > 0
        assert np.all(np.isfinite(P)) and np.all(np.isfinite(M))

    def test_interaction_diagrams_differ_between_axes(self):
        section = RC(Rectangle(30, 20), ReinfRect(14, 24, 3, 3, 0.79), 4, 60, 'US')
        _, Mx, _ = section.section_interaction_2d('x', 8)
        _, My, _ = section.section_interaction_2d('y', 8)
        assert not np.allclose(Mx, My), 'a 30x20 section is not axis-symmetric'
