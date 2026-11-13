"""Online recalibration — package facade.

The pre-split ``pipeline/recalibration.py`` became this package; the
facade re-exports its COMPLETE top-level namespace (underscore names
included) so every external reference — ``from .recalibration import
_build_controller``, the test suite's ``R.<name>`` reads — works
unchanged. tests/test_recal_facade_parity.py enforces this against the
frozen pre-split name list.

Layout: ``_constants`` (shared leaf) -> ``measurement`` (pure image/
geometry math) -> ``backends`` (per-lens-model measurement drivers) ->
``surfaces`` (panel + journal mixins) -> ``controller`` (the state
machine; assembles the mixins into ``RecalibrationController``).

Stays plain ``.py`` in compiled wheels (package re-export policy); the
six submodules compile individually per the manifest.
"""

import time  # noqa: F401 -- tests patch the shared time module via R.time

from ._constants import (  # noqa: F401
    _CHECK_EVERY,
    _CHECK_INTERVAL_S,
    _CHECK_INTERVAL_WARMUP_S,
    _BUFFER_SIZE,
    _POOL_GATE_PX,
    _WINDOW,
    _OK_PX,
    _THROTTLE_S,
    _Q_MIN_COV,
    _Q_MIN_N,
    _MAX_RECAL_ATTEMPTS,
    _PROBATION_CHECKS,
    _COOLDOWN_CHECKS,
    _WARMUP_CHECKS,
    _RANSAC_SEED,
    _MAX_ANGLE_DEG,
    _AZ_EDGE_FRAC,
    _V_MAX_ANGLE_PRIOR,
    _V_MIN_HOLDOUT,
    _V_MAX_USABLE_PX,
    _V_MIN_IMPROVE_PX,
    _V_MAX_DEPTH_DRIFT,
    _V_DEPTH_RANGE_M,
    _RECAL_QUEUE_MAX,
    _FEED_ACCEPT_INTERVAL_S,
    _FEED_REJECT_INTERVAL_S,
    _HISTORY_S,
    _MAX_SLOT_PTS,
    _STORE_SCHEMA,
    _STORE_MAX,
    _DHASH_SIDE,
    _MIN_NOVELTY_BITS,
    _RECAL_DRY_RUN,
    _prune_history,
    _throttle_sleep,
)
from .measurement import (  # noqa: F401
    sift_flann_match,
    orb_match,
    _GatePolicy,
    _GateResult,
    _gray_small,
    frame_gate,
    gate_pair,
    _scaled_K,
    _grid_coverage,
    epi_from_rectified_points,
    epi_on_rectified,
    _split_stereo,
    _dhash64,
    _hamming64,
    _build_rectify_maps,
    calculate_epipolar_error,
    _select_from_essential,
    _angle_between_R_deg,
    _relative_rotation_axes_deg,
    _average_rotations_svd,
    estimate_from_matches,
    estimate_with_prior,
    _RAYS_RANSAC_THR,
    estimate_with_prior_rays,
)
from .backends import (  # noqa: F401
    LatlonBundle,
    compose_new_R,
    _dR_between,
    _RecalBackend,
    _PinholeBackend,
    _FisheyeBackend,
)
from .surfaces import (  # noqa: F401
    _ObservabilityMixin,
)
from .storage import (  # noqa: F401
    _PersistenceMixin,
)
from .controller import (  # noqa: F401
    HealthMonitor,
    _ValidationPolicy,
    _ValidationResult,
    _validate,
    _State,
    RecalibrationStatus,
    RecalibrationController,
    _build_controller,
)
