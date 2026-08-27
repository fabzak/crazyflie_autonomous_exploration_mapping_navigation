"""Terrain-rejection datum lifecycle. Pure state machine, no dynamics, no ROS.

DESIGN-PROPOSAL for Architecture B. Not firmware, not built, not flashed.

Three SEPARATE concepts, never one boolean:
    detector_enabled      - is terrain anomaly logic running?
    datum_valid           - is there a trusted planar-floor reference?
    transition_active     - is a commanded layer change in progress?
A disabled detector does NOT mean the datum is invalid.
'Transition complete' does NOT mean 'new datum validated'.

The datum is a FLOOR CLEARANCE, never odom.z:
    clearance = down_range * Rzz        (attitude-corrected, valid geometry)
Across a commanded layer change the invariant is
    expected_new_clearance = old_valid_clearance + commanded_delta_z
where commanded_delta_z comes from the LAYER DEFINITION (layer_heights[n+1] -
layer_heights[n]), not from integrating a velocity command.
"""

# ---- PRE-DECLARED thresholds (locked before any test) -------------------
TERRAIN_THRESH = 0.015      # m, unchanged from the previous gate
REVALIDATE_TOL = 0.030      # m, |clearance - expected| to accept a new datum
STABLE_TOL = 0.010          # m, peak-to-peak over the stability window
N_STABLE = 20               # samples (0.5 s at 40 Hz)
REVALIDATE_DWELL = 1.0      # s
SETTLE_DWELL = 0.5          # s
VZ_CMD_EPS = 0.02           # m/s
VZ_EST_EPS = 0.05           # m/s
XY_EPS = 0.03               # m/s
TILT_MAX_DEG = 10.0
PHASE_TIMEOUT = 0.5         # s, matches the z_authority freshness contract
REVALIDATE_TIMEOUT = 5.0    # s
TRANSITION_TIMEOUT = 15.0   # s
ACQUIRE_TIMEOUT = 8.0       # s
T_ABORT = 3.0               # s, unchanged

HOLD_LAYER, VERTICAL_TRANSITION, LANDING, UNKNOWN = (
    'HOLD_LAYER', 'VERTICAL_TRANSITION', 'LANDING', 'UNKNOWN')
(UNINIT, DATUM_ACQUIRE, DATUM_VALID, XTRANSITION, SETTLING, DATUM_REVALIDATE,
 TERRAIN_SUSPECT, BRAKE, RETREAT, RECOVER, ABORT) = (
 'UNINIT', 'DATUM_ACQUIRE', 'DATUM_VALID', 'VERTICAL_TRANSITION', 'SETTLING',
 'DATUM_REVALIDATE', 'TERRAIN_SUSPECT', 'BRAKE', 'RETREAT', 'RECOVER', 'ABORT')

SUSPECT_STATES = (TERRAIN_SUSPECT, BRAKE, RETREAT)


class DatumLifecycle:
    def __init__(self):
        self.state = UNINIT
        self.datum_clearance = None      # the trusted planar-floor clearance
        self.expected_new = None         # pending clearance across a transition
        self.old_datum = None
        self.datum_epoch = 0
        self.t_state = 0.0
        self.t_stable = None
        self.window = []
        self.events = []
        self.rejected = 0
        self.transition_rejected = 0

    # ---- derived flags: three SEPARATE concepts ------------------------
    @property
    def detector_enabled(self):
        return self.state == DATUM_VALID

    @property
    def datum_valid(self):
        return self.state in (DATUM_VALID,) and self.datum_clearance is not None

    @property
    def transition_active(self):
        return self.state == XTRANSITION

    @property
    def tof_fuse(self):
        # Suppressed ONLY while terrain is suspect. During an intentional
        # transition over verified planar floor the ToF is still correct and
        # is the best short-horizon Z aid available (Mode 1, see report).
        return self.state not in SUSPECT_STATES

    @property
    def xy_allowed(self):
        return self.state in (DATUM_VALID,)

    @property
    def z_cmd_allowed(self):
        return self.state in (XTRANSITION, ABORT)

    def _go(self, s, t, why=''):
        if s != self.state:
            self.events.append((s, round(t, 3), why))
            self.state = s
            self.t_state = t
            self.t_stable = None
            self.window = []

    def _stable(self, clearance, t):
        self.window.append(clearance)
        if len(self.window) > N_STABLE:
            self.window.pop(0)
        if len(self.window) < N_STABLE:
            return False
        return (max(self.window) - min(self.window)) <= STABLE_TOL

    # ---- one sample ---------------------------------------------------
    def update(self, t, clearance, phase, phase_age, vz_cmd=0.0, vz_est=0.0,
               vxy=0.0, tilt_deg=0.0, delta_z=None, request_transition=False):
        """Returns the terrain-rejection decision for THIS ToF sample:
        True  = reject (do not let it reach the Kalman scalar update)."""
        geom_ok = (clearance is not None and tilt_deg <= TILT_MAX_DEG)
        # FAIL-SAFE: unknown or stale intent -> never arm, never explore.
        if phase_age > PHASE_TIMEOUT:
            phase = UNKNOWN

        st = self.state
        if st == ABORT:
            return False

        # A transition request is REJECTED while terrain is suspect.
        if request_transition and st in SUSPECT_STATES:
            self.transition_rejected += 1
            self.events.append(('TRANSITION_REJECTED', round(t, 3),
                                'terrain suspect'))

        if st == UNINIT:
            if geom_ok and phase == HOLD_LAYER:
                self._go(DATUM_ACQUIRE, t, 'geometry valid')
            return False

        if st == DATUM_ACQUIRE:
            if not geom_ok:
                self.window = []
                return False
            if self._stable(clearance, t):
                self.datum_clearance = sorted(self.window)[len(self.window)//2]
                self.datum_epoch += 1
                self._go(DATUM_VALID, t, f'datum={self.datum_clearance:.3f}')
            elif t - self.t_state > ACQUIRE_TIMEOUT:
                self._go(ABORT, t, 'datum acquisition timeout')
            return False

        if st == DATUM_VALID:
            if phase == VERTICAL_TRANSITION:
                self.old_datum = self.datum_clearance
                self.expected_new = (self.datum_clearance + delta_z
                                     if delta_z is not None else None)
                self._go(XTRANSITION, t,
                         f'expect {self.expected_new:.3f}'
                         if self.expected_new is not None else 'no delta')
                return False
            if phase in (UNKNOWN, LANDING):
                self._go(SETTLING if phase == UNKNOWN else ABORT, t,
                         'intent unknown' if phase == UNKNOWN else 'landing')
                return False
            if not geom_ok:
                return False                      # no decision without geometry
            if abs(clearance - self.datum_clearance) > TERRAIN_THRESH:
                self.rejected += 1
                self._go(TERRAIN_SUSPECT, t,
                         f'dev {clearance - self.datum_clearance:+.3f}')
                return True                        # FIRST-SAMPLE rejection
            return False

        if st == XTRANSITION:
            if phase == HOLD_LAYER:
                self._go(SETTLING, t, 'transition command ended')
            elif t - self.t_state > TRANSITION_TIMEOUT:
                self._go(ABORT, t, 'transition timeout')
            return False

        if st == SETTLING:
            quiet = (abs(vz_cmd) < VZ_CMD_EPS and abs(vz_est) < VZ_EST_EPS
                     and abs(vxy) < XY_EPS and tilt_deg <= TILT_MAX_DEG)
            if quiet:
                if self.t_stable is None:
                    self.t_stable = t
                elif t - self.t_stable >= SETTLE_DWELL:
                    self._go(DATUM_REVALIDATE, t, 'settled')
            else:
                self.t_stable = None
            if t - self.t_state > TRANSITION_TIMEOUT:
                self._go(ABORT, t, 'settling timeout')
            return False

        if st == DATUM_REVALIDATE:
            if not geom_ok:
                self.window = []; self.t_stable = None
            elif self._stable(clearance, t):
                # PRE-DECLARED REVALIDATE_DWELL: stability must be SUSTAINED,
                # not merely met once.  Latching on the first stable window
                # captured the settling tail and biased the new datum by
                # ~13 mm, which consumed almost the whole 15 mm terrain margin.
                if self.t_stable is None:
                    self.t_stable = t
                    return False
                if t - self.t_stable < REVALIDATE_DWELL:
                    return False
                median = sorted(self.window)[len(self.window)//2]
                if self.expected_new is None:
                    self._go(ABORT, t, 'no expected clearance')
                elif abs(median - self.expected_new) <= REVALIDATE_TOL:
                    self.datum_clearance = median
                    self.expected_new = None
                    self.datum_epoch += 1
                    self._go(DATUM_VALID, t,
                             f'revalidated {median:.3f} (epoch '
                             f'{self.datum_epoch})')
                else:
                    # STABLE BUT WRONG -> an obstacle, not the datum floor.
                    self._go(ABORT, t,
                             f'clearance {median:.3f} != expected '
                             f'{self.expected_new:.3f}: obstacle, not datum')
            if t - self.t_state > REVALIDATE_TIMEOUT:
                self._go(ABORT, t, 'revalidation timeout')
            return False

        if st == TERRAIN_SUSPECT:
            self._go(BRAKE, t, 'braking')
            return True
        if st == BRAKE:
            if abs(vxy) < XY_EPS:
                self._go(RETREAT, t, 'stopped')
            elif t - self.t_state > T_ABORT:
                self._go(ABORT, t, 'brake timeout')
            return True
        if st == RETREAT:
            if geom_ok and abs(clearance - self.datum_clearance) <= TERRAIN_THRESH:
                if self._stable(clearance, t):
                    self._go(RECOVER, t, 'datum geometry reacquired')
            else:
                self.window = []
            if t - self.t_state > T_ABORT:
                self._go(ABORT, t, 'fail-closed: datum not recovered')
            return True
        if st == RECOVER:
            if geom_ok and self._stable(clearance, t):
                self._go(DATUM_VALID, t, 'recovered')
            elif t - self.t_state > T_ABORT:
                self._go(ABORT, t, 'recovery timeout')
            return False
        return False
