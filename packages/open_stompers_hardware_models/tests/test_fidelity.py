"""Fidelity tests: LinkForge models vs the upstream URDF fixtures.

Two layers of comparison:

* ``test_model_matches_upstream`` compares the in-memory LinkForge model against
  the upstream URDF parsed by LinkForge, field by field. This is exact (both
  sides come from the same decimal literals).
* ``test_export_roundtrip`` exports our model back to URDF and re-parses it,
  verifying the generator preserves the model. LinkForge formats exported
  floats with six decimal places, so this comparison uses a matching tolerance.
"""

import math
from collections.abc import Callable
from typing import Final

from linkforge.core import (
    Box,
    Cylinder,
    Geometry,
    Joint,
    JointType,
    Link,
    Mesh,
    Robot,
    Sphere,
    Transform,
    URDFGenerator,
    Vector3,
    read_urdf,
)

from open_stompers_hardware_models import load
from open_stompers_hardware_models.robots.mini_pi_plus import MiniPiPlusVariant

TOL: Final[float] = 1e-15
ROUNDTRIP_TOL: Final[float] = 5.1e-7

FIRST_VARIANT: Final[MiniPiPlusVariant] = (
    MiniPiPlusVariant.TWELVE_LEG_TEN_ARM_WITHOUT_GRIPPERS_WITHOUT_HEAD_WITH_WAIST
)
CONTINUOUS_OVERRIDES: Final[dict[MiniPiPlusVariant, frozenset[str]]] = {
    FIRST_VARIANT: frozenset({"l_wrist_joint", "r_wrist_joint"}),
}


def _close(a: float | int | None, b: float | int | None, tol: float) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _assert_vector_close(expected: Vector3, actual: Vector3, tol: float) -> None:
    assert _close(expected.x, actual.x, tol)
    assert _close(expected.y, actual.y, tol)
    assert _close(expected.z, actual.z, tol)


def _assert_geometry_equal(
    expected: Geometry, actual: Geometry, tol: float, map_resource: Callable[[str], str]
) -> None:
    assert type(expected) is type(actual), (
        f"geometry type mismatch: {type(expected).__name__} vs {type(actual).__name__}"
    )
    if isinstance(expected, Box) and isinstance(actual, Box):
        _assert_vector_close(expected.size, actual.size, tol)
    elif isinstance(expected, Cylinder) and isinstance(actual, Cylinder):
        assert _close(expected.radius, actual.radius, tol)
        assert _close(expected.length, actual.length, tol)
    elif isinstance(expected, Sphere) and isinstance(actual, Sphere):
        assert _close(expected.radius, actual.radius, tol)

    elif isinstance(expected, Mesh) and isinstance(actual, Mesh):
        assert map_resource(expected.resource) == actual.resource
        _assert_vector_close(expected.scale, actual.scale, tol)


def _assert_origin_close(expected: Transform, actual: Transform, tol: float) -> None:
    _assert_vector_close(expected.xyz, actual.xyz, tol)
    _assert_vector_close(expected.rpy, actual.rpy, tol)


def _assert_joint_equal(
    expected: Joint, actual: Joint, tol: float, continuous_overrides: frozenset[str]
) -> None:
    assert expected.name == actual.name
    assert expected.parent == actual.parent
    assert expected.child == actual.child

    if expected.name in continuous_overrides:
        assert expected.type is JointType.REVOLUTE
        assert actual.type is JointType.CONTINUOUS
    else:
        assert expected.type is actual.type

    _assert_origin_close(expected.origin, actual.origin, tol)

    if actual.type is JointType.FIXED:
        assert actual.axis is None
        assert actual.limits is None
        # LinkForge drops dynamics for fixed joints; upstream may carry them.
        return

    assert expected.axis is not None
    assert actual.axis is not None
    _assert_vector_close(expected.axis, actual.axis, tol)

    if expected.name in continuous_overrides:
        assert actual.limits is None
    elif actual.limits is None:
        assert expected.limits is None
    else:
        assert expected.limits is not None
        assert _close(expected.limits.lower, actual.limits.lower, tol)
        assert _close(expected.limits.upper, actual.limits.upper, tol)
        assert _close(expected.limits.effort, actual.limits.effort, tol)
        assert _close(expected.limits.velocity, actual.limits.velocity, tol)

    if expected.dynamics is None:
        assert actual.dynamics is None
    else:
        assert actual.dynamics is not None
        assert _close(expected.dynamics.damping, actual.dynamics.damping, tol)
        assert _close(expected.dynamics.friction, actual.dynamics.friction, tol)


def _assert_link_equal(
    expected: Link,
    actual: Link,
    tol: float,
    map_resource: Callable[[str], str],
    compare_materials: bool = True,
) -> None:
    assert expected.name == actual.name

    expected_inertial = expected.inertial
    actual_inertial = actual.inertial
    assert expected_inertial is not None
    assert actual_inertial is not None
    assert _close(expected_inertial.mass, actual_inertial.mass, tol)
    _assert_origin_close(expected_inertial.origin, actual_inertial.origin, tol)
    expected_i = expected_inertial.inertia
    actual_i = actual_inertial.inertia
    for field, expected_value, actual_value in (
        ("ixx", expected_i.ixx, actual_i.ixx),
        ("iyy", expected_i.iyy, actual_i.iyy),
        ("izz", expected_i.izz, actual_i.izz),
        ("ixy", expected_i.ixy, actual_i.ixy),
        ("ixz", expected_i.ixz, actual_i.ixz),
        ("iyz", expected_i.iyz, actual_i.iyz),
    ):
        assert _close(expected_value, actual_value, tol), f"{actual.name}: inertia {field} mismatch"

    assert len(expected.visuals) == len(actual.visuals)
    for expected_visual, actual_visual in zip(expected.visuals, actual.visuals, strict=True):
        _assert_origin_close(expected_visual.origin, actual_visual.origin, tol)
        _assert_geometry_equal(expected_visual.geometry, actual_visual.geometry, tol, map_resource)
        if compare_materials:
            assert (expected_visual.material is None) == (actual_visual.material is None)
            if expected_visual.material is None or actual_visual.material is None:
                continue
            expected_color = expected_visual.material.color
            actual_color = actual_visual.material.color
            assert expected_color is not None
            assert actual_color is not None
            for expected_value, actual_value in (
                (expected_color.r, actual_color.r),
                (expected_color.g, actual_color.g),
                (expected_color.b, actual_color.b),
                (expected_color.a, actual_color.a),
            ):
                assert _close(expected_value, actual_value, tol)

    assert len(expected.collisions) == len(actual.collisions)
    for expected_col, actual_col in zip(expected.collisions, actual.collisions, strict=True):
        _assert_origin_close(expected_col.origin, actual_col.origin, tol)
        _assert_geometry_equal(expected_col.geometry, actual_col.geometry, tol, map_resource)


def _assert_same_robot(
    expected: Robot,
    actual: Robot,
    tol: float,
    map_resource: Callable[[str], str],
    continuous_overrides: frozenset[str] = frozenset(),
    compare_materials: bool = True,
) -> None:
    assert set(expected.graph.link_names) == set(actual.graph.link_names)
    expected_joints = {joint.name: joint for joint in expected.graph.joints}
    actual_joints = {joint.name: joint for joint in actual.graph.joints}
    assert set(expected_joints) == set(actual_joints)

    for name in expected.graph.link_names:
        expected_link = expected.get_link(name)
        actual_link = actual.get_link(name)
        assert expected_link is not None
        assert actual_link is not None
        _assert_link_equal(
            expected_link,
            actual_link,
            tol,
            map_resource,
            compare_materials=compare_materials,
        )
    for name, expected_joint in expected_joints.items():
        _assert_joint_equal(expected_joint, actual_joints[name], tol, continuous_overrides)


def _map_upstream_resource(variant: MiniPiPlusVariant) -> Callable[[str], str]:
    def _map(resource: str) -> str:
        filename = resource.rsplit("/", 1)[-1]
        return f"hardware_models/mini_pi_plus/{variant}/meshes/{filename}"

    return _map


def test_model_matches_upstream(variant: MiniPiPlusVariant, upstream_robot: Robot) -> None:
    ours = load(variant)
    _assert_same_robot(
        upstream_robot,
        ours,
        tol=TOL,
        map_resource=_map_upstream_resource(variant),
        continuous_overrides=CONTINUOUS_OVERRIDES.get(variant, frozenset()),
        compare_materials=False,
    )
    assert _close(upstream_robot.total_mass, ours.total_mass, TOL)


def test_export_roundtrip(variant: MiniPiPlusVariant) -> None:
    ours = load(variant)
    reparsed = read_urdf(URDFGenerator().generate(ours))
    assert reparsed.name == variant
    _assert_same_robot(
        ours,
        reparsed,
        tol=ROUNDTRIP_TOL,
        map_resource=lambda resource: resource,
    )
    assert _close(ours.total_mass, reparsed.total_mass, ROUNDTRIP_TOL)
