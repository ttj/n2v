"""Top-level pytest configuration shared across every test subdirectory.

The set-validity assertion helpers are monkeypatched onto the ``pytest``
module so tests can call ``pytest.assert_star_valid(...)`` etc. They live
here (the rootdir conftest) rather than only under ``tests/unit/`` so they
are available to EVERY invocation — ``pytest tests/unit/``,
``pytest tests/integration/``, and ``pytest tests/soundness/`` alike.

CI runs those three directories as separate ``pytest`` commands, so a
helper registered only under ``tests/unit/`` is absent when
``tests/integration/`` runs on its own (a full ``pytest tests/`` run
happens to work because ``tests/unit/conftest.py`` loads first and sets
the attribute process-wide). Registering here closes that gap.

NOTE: ``tests/unit/conftest.py`` (and the ``layer_ops`` / ``sets`` unit
conftests) still define their own copies of these helpers. Those are now
redundant with this file; consolidating them is a safe follow-up cleanup.
"""

import pytest


def assert_star_valid(star):
    """Assert that a Star set is valid."""
    assert star.V is not None
    assert star.C is not None
    assert star.d is not None
    assert star.V.shape[1] == star.nVar + 1
    assert star.C.shape[1] == star.nVar


def assert_zono_valid(zono):
    """Assert that a Zonotope is valid."""
    assert zono.c is not None
    assert zono.V is not None
    assert len(zono.c.shape) == 2
    assert zono.c.shape[1] == 1


def assert_image_star_valid(img_star):
    """Assert that an ImageStar is valid."""
    # ImageStar has 4D V: (H, W, C, nVar+1), so we check differently than Star
    assert img_star.V is not None
    assert img_star.C is not None
    assert img_star.d is not None
    assert img_star.V.ndim == 4, f"ImageStar V should be 4D, got {img_star.V.ndim}D"
    assert img_star.V.shape[3] == img_star.nVar + 1
    assert img_star.C.shape[1] == img_star.nVar
    assert img_star.height > 0
    assert img_star.width > 0
    assert img_star.num_channels > 0
    assert img_star.V.shape[0] == img_star.height
    assert img_star.V.shape[1] == img_star.width
    assert img_star.V.shape[2] == img_star.num_channels
    assert img_star.dim == img_star.height * img_star.width * img_star.num_channels


def assert_hexatope_valid(hexatope):
    """Assert that a Hexatope is valid."""
    assert hexatope.center is not None
    assert hexatope.generators is not None
    assert hexatope.dcs is not None
    assert hexatope.generators.shape[0] == hexatope.dim
    assert hexatope.generators.shape[1] == hexatope.nVar


def assert_octatope_valid(octatope):
    """Assert that an Octatope is valid."""
    assert octatope.center is not None
    assert octatope.generators is not None
    assert octatope.utvpi is not None
    assert octatope.generators.shape[0] == octatope.dim
    assert octatope.generators.shape[1] == octatope.nVar


# Make helpers available via the pytest module namespace.
pytest.assert_star_valid = assert_star_valid
pytest.assert_zono_valid = assert_zono_valid
pytest.assert_image_star_valid = assert_image_star_valid
pytest.assert_hexatope_valid = assert_hexatope_valid
pytest.assert_octatope_valid = assert_octatope_valid
