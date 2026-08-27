#!/usr/bin/env python3
"""Shared real-hardware operator supervisor for ``layer_explore_real`` and
``cf_auto_real``.

Launching a real profile must never arm the aircraft or start autonomous
motion.  This node owns the only path from a human keypress to an arm, start,
land or emergency request, and it publishes the single authorization heartbeat
that both the algorithm start gate and the control adapter require.

Keys (see :class:`KeyEvent`)::

    Alt     ARM / ground DISARM
    G       START AUTONOMY
    L       ABORT + CONTROLLED LAND
    SPACE   EMERGENCY MOTOR STOP

The decision logic lives in :class:`OperatorSupervisor`, which is ROS-free and
driven by caller-supplied monotonic times so every interlock is unit-testable
without hardware.  The ROS node is an I/O shell: keyboard acquisition, service
calls and the authorization heartbeat.

Fail-closed everywhere.  Authorization is a *heartbeat*, never a latch: if this
node dies, stalls, or its keyboard backend fails, the heartbeat stops, the
control adapter's freshness gate expires, and an airborne vehicle is landed by
the adapter's existing notify/land handover.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

try:
    import rclpy
    from crazyflie_interfaces.msg import Status
    from crazyflie_interfaces.srv import Arm, Land
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from std_msgs.msg import Bool, String
    from std_srvs.srv import Empty
    ROS_AVAILABLE = True
except ModuleNotFoundError:  # Pure supervisor stays importable without ROS.
    rclpy = None
    Node = object
    ROS_AVAILABLE = False


CONTROL_LEGEND = (
    '============================================================\n'
    'REAL CRAZYFLIE OPERATOR CONTROL\n'
    '  Alt   : ARM / ground DISARM\n'
    '  G     : START AUTONOMY\n'
    '  L     : ABORT + CONTROLLED LAND\n'
    '  SPACE : EMERGENCY MOTOR STOP\n'
    '============================================================')


class KeyEvent(str, Enum):
    ARM_TOGGLE = 'ARM_TOGGLE'      # Alt
    START = 'START'                # G
    LAND = 'LAND'                  # L
    EMERGENCY = 'EMERGENCY'        # Space


class OperatorState(str, Enum):
    DISARMED = 'DISARMED'
    ARMING_OPERATOR = 'ARMING_OPERATOR'
    ARMED_IDLE = 'ARMED_IDLE'
    AUTONOMY_RUNNING = 'AUTONOMY_RUNNING'
    LANDING = 'LANDING'
    DISARMED_STOPPED = 'DISARMED_STOPPED'
    EMERGENCY_LATCHED = 'EMERGENCY_LATCHED'


#: States in which the operator has released the autonomy gate.
AUTHORIZED_STATES = (OperatorState.AUTONOMY_RUNNING,)

#: crazyflie_interfaces/msg/Status supervisor bits, repeated so the pure
#: supervisor never needs a ROS message to make a decision.
SUPERVISOR_CAN_BE_ARMED = 1
SUPERVISOR_IS_ARMED = 2
SUPERVISOR_CAN_FLY = 8
SUPERVISOR_IS_FLYING = 16
SUPERVISOR_IS_TUMBLED = 32
SUPERVISOR_IS_LOCKED = 64


@dataclass(frozen=True)
class OperatorConfig:
    """Freshness and confirmation bounds for :class:`OperatorSupervisor`."""

    status_timeout_sec: float = 1.50
    odom_timeout_sec: float = 0.50
    arm_confirmation_timeout_sec: float = 3.0
    disarm_confirmation_timeout_sec: float = 3.0
    #: How long after a confirmed touchdown the run is considered finished.
    land_confirmation_timeout_sec: float = 30.0
    #: How far above its RESTING altitude the vehicle must rise before it is
    #: treated as airborne, used only together with a supervisor that is not
    #: reporting IS_FLYING.  This is a margin, not an absolute altitude: a
    #: Crazyflie taking off from a 0.16 m stand rests above any sensible
    #: absolute threshold, and an absolute test latched it permanently
    #: airborne so it could never be armed (observed 2026-08-22).  The
    #: resting altitude is learned while the supervisor reports not-flying.
    grounded_height_m: float = 0.12
    #: How long continuous grounded evidence must hold before the airborne
    #: latch is released.  The firmware IS_FLYING bit is the real protection;
    #: this debounce only stops a momentary altitude dip from clearing it.
    grounded_debounce_sec: float = 1.0
    #: Largest planar distance from the odometry origin at which autonomy may
    #: be started, in metres.  The Flow deck integrates apparent motion while
    #: the airframe is hand-carried, and the estimate does not reset until the
    #: next boot, so a drone moved after power-up can report tens of metres.
    #: layer_explore maps into a fixed +/-16.25 m grid centred on that origin,
    #: so starting from a drifted pose maps off-grid from the first sample.
    #: 0 disables the check.
    max_start_offset_m: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.status_timeout_sec, self.odom_timeout_sec,
            self.arm_confirmation_timeout_sec,
            self.disarm_confirmation_timeout_sec,
            self.land_confirmation_timeout_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError('operator timeouts must be finite')
        if any(value <= 0.0 for value in values):
            raise ValueError('operator timeouts must be > 0')
        if (not math.isfinite(self.grounded_height_m)
                or self.grounded_height_m < 0.0):
            raise ValueError('grounded height must be finite and >= 0')
        if (not math.isfinite(self.grounded_debounce_sec)
                or self.grounded_debounce_sec < 0.0):
            raise ValueError('grounded debounce must be finite and >= 0')
        if (not math.isfinite(self.max_start_offset_m)
                or self.max_start_offset_m < 0.0):
            raise ValueError('max start offset must be finite and >= 0')


@dataclass(frozen=True)
class OperatorDecision:
    """One evaluation result: what to publish and what to say about it."""

    state: OperatorState
    authorized: bool
    message: str = ''


class OperatorSupervisor:
    """Pure operator interlock state machine.

    Every method takes a monotonic ``now`` supplied by the caller.  No wall
    clock, no ROS, no I/O.  The only outputs are the current state, the
    authorization flag, and :meth:`required_service`.
    """

    def __init__(self, config: OperatorConfig = OperatorConfig()):
        self.config = config
        self.state = OperatorState.DISARMED
        self.state_since = 0.0
        self.emergency_latched = False
        self.land_latched = False

        self._supervisor = 0
        self._status_at: Optional[float] = None
        self._height: Optional[float] = None
        self._planar_offset: Optional[float] = None
        self._odom_at: Optional[float] = None
        #: Set by any evidence the vehicle has left the ground, and cleared
        #: only by sustained positive evidence that it is back on it.  A
        #: vehicle that was picked up by hand and set down again must be
        #: armable, but a momentary altitude dip in flight must not clear it.
        self._airborne_latched = False
        self._grounded_since: Optional[float] = None
        #: Altitude the vehicle reads while resting on whatever it sits on.
        self._ground_reference_z: Optional[float] = None

        self.arm_request_pending = False
        self.disarm_request_pending = False
        self.land_request_pending = False
        self.emergency_request_pending = False
        self._completed_service: Optional[str] = None

    # ── telemetry ─────────────────────────────────────────────────────────

    def update_status(self, supervisor_info: int, now: float) -> None:
        if not math.isfinite(now):
            return
        self._supervisor = int(supervisor_info)
        self._status_at = now
        if self._supervisor & SUPERVISOR_IS_FLYING:
            self._airborne_latched = True

    def update_odometry(self, height: float, now: float,
                        x: float = 0.0, y: float = 0.0) -> None:
        if not (math.isfinite(height) and math.isfinite(now)):
            return
        self._height = float(height)
        self._planar_offset = (math.hypot(x, y)
                               if math.isfinite(x) and math.isfinite(y)
                               else None)
        self._odom_at = now
        settled = (self.status_fresh(now)
                   and not self._supervisor & SUPERVISOR_IS_FLYING
                   and not self._airborne_latched)
        if self._ground_reference_z is None:
            # First reading: learn where "on the ground" is for this setup,
            # whether that is the floor or a raised stand.
            if settled:
                self._ground_reference_z = float(height)
            return
        if height > self.airborne_height_threshold():
            # Checked BEFORE any further learning, so a climbing aircraft can
            # never drag its own reference upward and hide the climb.
            self._airborne_latched = True
        elif settled:
            self._ground_reference_z = float(height)

    def airborne_height_threshold(self) -> float:
        """Altitude above which the vehicle is treated as off its support."""
        reference = (self._ground_reference_z
                     if self._ground_reference_z is not None else 0.0)
        return reference + self.config.grounded_height_m

    def _fresh(self, stamp: Optional[float], timeout: float,
               now: float) -> bool:
        if stamp is None or not math.isfinite(now):
            return False
        age = now - stamp
        return 0.0 <= age <= timeout

    def status_fresh(self, now: float) -> bool:
        return self._fresh(self._status_at, self.config.status_timeout_sec, now)

    def odometry_fresh(self, now: float) -> bool:
        return self._fresh(self._odom_at, self.config.odom_timeout_sec, now)

    def confirmed_armed(self, now: float) -> bool:
        """Only a fresh supervisor report may assert ARMED."""
        return bool(self.status_fresh(now)
                    and self._supervisor & SUPERVISOR_IS_ARMED)

    def airborne(self, now: float) -> bool:
        """Whether the vehicle must be assumed to be off the ground.

        Latched on the first evidence of flight and cleared only by a
        confirmed landing.  Unknown counts as airborne: a stale supervisor
        can never be used to justify cutting the motors.
        """
        if self._airborne_latched:
            return True
        if not self.status_fresh(now):
            return True
        return bool(self._supervisor & SUPERVISOR_IS_FLYING)

    def confirmed_grounded(self, now: float) -> bool:
        """Positive, fresh evidence that the vehicle is on the ground."""
        if self.emergency_latched:
            return False
        if not self.status_fresh(now):
            return False
        if self._supervisor & SUPERVISOR_IS_FLYING:
            return False
        if self._airborne_latched:
            # Once it has flown, require a fresh low altitude as well.
            if not self.odometry_fresh(now) or self._height is None:
                return False
            return self._height <= self.airborne_height_threshold()
        return True

    # ── operator events ───────────────────────────────────────────────────

    def _set_state(self, state: OperatorState, now: float) -> None:
        if state == self.state:
            return
        self.state = state
        self.state_since = now

    def on_key(self, event: KeyEvent, now: float) -> str:
        """Apply one debounced operator keypress; return a log message."""
        if event == KeyEvent.EMERGENCY:
            return self._on_emergency(now)
        if self.emergency_latched:
            return (f'{event.value} REJECTED: emergency stop is latched; '
                    'Crazyflie firmware reset and relaunch are required')
        if event == KeyEvent.ARM_TOGGLE:
            return self._on_arm_toggle(now)
        if event == KeyEvent.START:
            return self._on_start(now)
        if event == KeyEvent.LAND:
            return self._on_land(now)
        return f'unknown operator event {event}'

    def _on_emergency(self, now: float) -> str:
        if self.emergency_latched:
            return 'EMERGENCY STOP already latched'
        self.emergency_latched = True
        self.land_latched = True
        self.arm_request_pending = False
        self.disarm_request_pending = False
        self.land_request_pending = False
        self.emergency_request_pending = True
        self._set_state(OperatorState.EMERGENCY_LATCHED, now)
        return ('EMERGENCY STOP ACTIVATED - motor power cut; Crazyflie '
                'firmware reset required before any further flight')

    def _on_arm_toggle(self, now: float) -> str:
        if self.state == OperatorState.DISARMED_STOPPED:
            return ('ARM REJECTED: this run has ended after an operator '
                    'land; relaunch before flying again')
        if self.state in (OperatorState.AUTONOMY_RUNNING,
                          OperatorState.LANDING):
            return ('DISARM REJECTED: aircraft is airborne - use L to land '
                    'or SPACE for emergency stop')
        if self.state == OperatorState.ARMING_OPERATOR:
            return 'ARM REQUEST ignored: an arm request is already in flight'
        if self.state == OperatorState.DISARMED:
            if not self.status_fresh(now):
                return ('ARM REJECTED: Crazyflie status is stale or absent; '
                        'cannot confirm the vehicle is safe to arm')
            if self._supervisor & (SUPERVISOR_IS_TUMBLED
                                   | SUPERVISOR_IS_LOCKED):
                return ('ARM REJECTED: supervisor reports the vehicle '
                        'tumbled or locked')
            if self.airborne(now):
                return ('ARM REJECTED: aircraft is airborne or its state is '
                        'unknown')
            if self._supervisor & SUPERVISOR_IS_ARMED:
                # The vehicle survived a previous run already armed, so
                # CAN_BE_ARMED is clear and a fresh Arm request would be
                # refused.  Adopt the real vehicle state instead of
                # deadlocking; the operator still pressed a key to get here,
                # and a second press now disarms as usual.
                self._set_state(OperatorState.ARMED_IDLE, now)
                return ('KEY Alt -> vehicle is already ARMED; adopting '
                        'ARMED_IDLE. Press Alt again to disarm.')
            if not self._supervisor & SUPERVISOR_CAN_BE_ARMED:
                return 'ARM REJECTED: supervisor reports CAN_BE_ARMED clear'
            self.arm_request_pending = True
            self._set_state(OperatorState.ARMING_OPERATOR, now)
            return 'KEY Alt -> ARM REQUEST'
        # ARMED_IDLE: ground disarm is permitted, airborne disarm never is.
        if not self.confirmed_grounded(now):
            return ('DISARM REJECTED: aircraft is airborne - use L to land '
                    'or SPACE for emergency stop')
        self.disarm_request_pending = True
        return 'KEY Alt -> GROUND DISARM REQUEST'

    def start_blockers(self, now: float) -> Tuple[str, ...]:
        """Every unmet precondition for releasing the autonomy gate."""
        blockers = []
        if self.emergency_latched:
            blockers.append('emergency latched')
        if self.land_latched:
            blockers.append('operator land latched for this run')
        if self.state == OperatorState.DISARMED:
            blockers.append('drone is not armed. Press Alt to arm first')
        elif self.state == OperatorState.ARMING_OPERATOR:
            blockers.append('arming has not been confirmed yet')
        elif self.state != OperatorState.ARMED_IDLE:
            blockers.append(f'operator state is {self.state.value}')
        if not self.status_fresh(now):
            blockers.append('Crazyflie status stale or absent')
        elif not self._supervisor & SUPERVISOR_IS_ARMED:
            blockers.append('supervisor does not report ARMED')
        elif self._supervisor & (SUPERVISOR_IS_TUMBLED | SUPERVISOR_IS_LOCKED):
            blockers.append('supervisor reports tumbled or locked')
        elif not self._supervisor & SUPERVISOR_CAN_FLY:
            blockers.append('supervisor reports CAN_FLY clear')
        if not self.odometry_fresh(now):
            blockers.append('odometry stale or absent')
        elif self.config.max_start_offset_m > 0.0:
            if self._planar_offset is None:
                blockers.append('odometry position is not finite')
            elif self._planar_offset > self.config.max_start_offset_m:
                blockers.append(
                    f'odometry position is {self._planar_offset:.1f} m from '
                    f'the origin (limit {self.config.max_start_offset_m:.1f} '
                    'm); the estimator has drifted, power-cycle the Crazyflie '
                    'where it will take off and do not carry it afterwards')
        if self.airborne(now):
            blockers.append('aircraft is not in the expected pre-takeoff state')
        return tuple(blockers)

    def _on_start(self, now: float) -> str:
        if self.state == OperatorState.AUTONOMY_RUNNING:
            return 'START ignored: autonomy is already running'
        blockers = self.start_blockers(now)
        if blockers:
            return 'START REJECTED: ' + '; '.join(blockers)
        self._set_state(OperatorState.AUTONOMY_RUNNING, now)
        return 'KEY g -> START REQUEST accepted; autonomy authorized'

    def _on_land(self, now: float) -> str:
        if self.state == OperatorState.LANDING:
            return 'LAND ignored: a controlled landing is already in progress'
        if self.state in (OperatorState.DISARMED,
                          OperatorState.DISARMED_STOPPED):
            return 'LAND ignored: the aircraft is not flying'
        # Authorization is revoked before any service call is made, so no
        # autonomous command can be forwarded during the handover.
        self.land_latched = True
        self.land_request_pending = True
        self._set_state(OperatorState.LANDING, now)
        return 'KEY l -> CONTROLLED LAND REQUEST; autonomy authorization revoked'

    # ── outputs ───────────────────────────────────────────────────────────

    def authorized(self, now: float) -> bool:
        """Whether autonomous motion is authorized on this exact tick."""
        if self.emergency_latched or self.land_latched:
            return False
        if self.state not in AUTHORIZED_STATES:
            return False
        # Authorization is only ever asserted against fresh evidence.
        return bool(self.status_fresh(now) and self.odometry_fresh(now))

    def required_service(self) -> Optional[str]:
        """The one hardware request that must be issued next, if any."""
        if self.emergency_request_pending:
            return 'emergency'
        if self.land_request_pending:
            return 'land'
        if self.arm_request_pending:
            return 'arm'
        if self.disarm_request_pending:
            return 'disarm'
        return None

    def service_result(self, operation: str, success: bool,
                       now: float) -> str:
        """Record one completed hardware request."""
        if operation != self.required_service():
            return ''
        self._completed_service = operation
        if operation == 'emergency':
            self.emergency_request_pending = False
            return ('emergency stop delivered'
                    if success else
                    'EMERGENCY SERVICE FAILED - cut power at the battery')
        if operation == 'land':
            self.land_request_pending = False
            return ('controlled landing commanded'
                    if success else
                    'LAND SERVICE FAILED - press SPACE if the aircraft is '
                    'not descending')
        if operation == 'arm':
            self.arm_request_pending = False
            if not success:
                self._set_state(OperatorState.DISARMED, now)
                return 'ARM FAILED: service call rejected; still DISARMED'
            # Acceptance is not confirmation.  tick() waits for IS_ARMED.
            return 'arm service accepted; waiting for supervisor confirmation'
        if operation == 'disarm':
            self.disarm_request_pending = False
            return ('ground disarm commanded'
                    if success else 'DISARM FAILED: service call rejected')
        return ''

    def _update_grounded_evidence(self, now: float) -> None:
        """Release the airborne latch on sustained grounded evidence.

        The firmware IS_FLYING bit does the real work: it stays set for the
        whole flight, so no in-flight altitude dip can clear the latch.  The
        debounce only guards against a single stale-looking sample.
        """
        grounded_now = (
            self.status_fresh(now)
            and not self._supervisor & SUPERVISOR_IS_FLYING
            and self.odometry_fresh(now)
            and self._height is not None
            and self._height <= self.airborne_height_threshold())
        if not grounded_now:
            self._grounded_since = None
            return
        if self._grounded_since is None:
            self._grounded_since = now
        elif (self._airborne_latched
              and now - self._grounded_since
              >= self.config.grounded_debounce_sec):
            self._airborne_latched = False

    def tick(self, now: float) -> OperatorDecision:
        """Advance confirmation-driven transitions."""
        if not math.isfinite(now):
            return OperatorDecision(self.state, False, '')
        # LANDING owns the latch itself: it must reach DISARMED_STOPPED on a
        # confirmed touchdown rather than silently un-latching beforehand.
        if self.state != OperatorState.LANDING:
            self._update_grounded_evidence(now)
        message = ''

        if self.state == OperatorState.ARMING_OPERATOR:
            if self.confirmed_armed(now):
                self._set_state(OperatorState.ARMED_IDLE, now)
                message = ('OPERATOR STATE: ARMED_IDLE - press G to start')
            elif (now - self.state_since
                  > self.config.arm_confirmation_timeout_sec):
                self.arm_request_pending = False
                self._set_state(OperatorState.DISARMED, now)
                message = ('ARM NOT CONFIRMED before timeout; back to '
                           'DISARMED')
        elif self.state == OperatorState.ARMED_IDLE:
            if not self.confirmed_armed(now) and not self.airborne(now):
                # A ground disarm, a link loss, or a firmware auto-disarm.
                self._set_state(OperatorState.DISARMED, now)
                message = 'OPERATOR STATE: DISARMED (no confirmed ARMED)'
        elif self.state == OperatorState.AUTONOMY_RUNNING:
            if self.emergency_latched:
                self._set_state(OperatorState.EMERGENCY_LATCHED, now)
            elif self.confirmed_grounded(now) and not self.confirmed_armed(now):
                # The run ended without an operator L: a watchdog latch, an
                # adapter fault landing, a supervisor cannot-fly, or an
                # externally completed landing brought the aircraft down and
                # the firmware disarmed it.  Observed 2026-08-23: the adapter
                # landed and disarmed on stale:odom:source_stamp while this
                # node still reported AUTONOMY_RUNNING for a disarmed aircraft
                # sitting on the floor.  Converge on the same terminal state
                # an operator landing reaches.
                #
                # Both predicates demand fresh status, so a stale tick blocks
                # this rather than triggering it, and it cannot fire before
                # takeoff because the vehicle is ARMED there.  DISARMED_STOPPED
                # is terminal, so this can never re-arm or restart autonomy.
                self._airborne_latched = False
                self._set_state(OperatorState.DISARMED_STOPPED, now)
                message = ('OPERATOR STATE: DISARMED_STOPPED - autonomy ended '
                           'without an operator land; aircraft is grounded '
                           'and disarmed. Relaunch before flying again.')
        elif self.state == OperatorState.LANDING:
            if self.confirmed_grounded(now):
                self._airborne_latched = False
                self._set_state(OperatorState.DISARMED_STOPPED, now)
                message = ('OPERATOR STATE: DISARMED_STOPPED - landing '
                           'confirmed; relaunch before flying again')
            elif (now - self.state_since
                  > self.config.land_confirmation_timeout_sec):
                message = ('LANDING NOT CONFIRMED within '
                           f'{self.config.land_confirmation_timeout_sec:.0f} s'
                           ' - press SPACE if the aircraft is still flying')
                self.state_since = now
        elif self.state == OperatorState.DISARMED_STOPPED:
            # Never leave a brushless airframe armed on the floor.
            if (self.confirmed_armed(now)
                    and not self.disarm_request_pending
                    and self._completed_service != 'disarm'):
                self.disarm_request_pending = True
                message = ('post-landing disarm requested; motors must not '
                           'stay live on the ground')
        return OperatorDecision(self.state, self.authorized(now), message)


# ── keyboard acquisition ──────────────────────────────────────────────────


#: Mapping from a backend-neutral key name to the operator event it raises.
KEY_BINDINGS = {
    'alt': KeyEvent.ARM_TOGGLE,
    'g': KeyEvent.START,
    'l': KeyEvent.LAND,
    'space': KeyEvent.EMERGENCY,
}


class KeyDebouncer:
    """Turn raw press/release streams into exactly one event per press.

    X11 auto-repeat delivers a continuous press stream for a held key, which
    would otherwise re-issue an arm, land or emergency request many times a
    second.  A key must be released before it can fire again.
    """

    def __init__(self):
        self._down = set()

    def press(self, key: str) -> Optional[KeyEvent]:
        if key not in KEY_BINDINGS or key in self._down:
            return None
        self._down.add(key)
        return KEY_BINDINGS[key]

    def release(self, key: str) -> None:
        self._down.discard(key)

    def clear(self) -> None:
        self._down.clear()


def _pynput_key_name(key) -> Optional[str]:
    """Normalise one pynput key object to a binding name."""
    from pynput import keyboard as pynput_keyboard

    # Left and right Alt both arm.  Key.alt_gr is deliberately NOT bound:
    # on layouts that have it, AltGr is a character-composition modifier
    # pressed while typing ordinary symbols, and arming on it would fire
    # the critical toggle during normal typing.
    if key in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r,
               pynput_keyboard.Key.alt):
        return 'alt'
    if key == pynput_keyboard.Key.space:
        return 'space'
    character = getattr(key, 'char', None)
    if isinstance(character, str) and character:
        lowered = character.lower()
        if lowered in ('g', 'l'):
            return lowered
    # Ctrl+G / Ctrl+L arrive as control characters; treat them as the letter.
    if isinstance(character, str) and len(character) == 1:
        code = ord(character)
        if code == 7:
            return 'g'
        if code == 12:
            return 'l'
    return None


class PynputKeyboardBackend:
    """Global X11 key hook.  No elevated privileges, no /dev/input access."""

    name = 'pynput'

    def __init__(self, sink: 'queue.Queue'):
        from pynput import keyboard as pynput_keyboard

        self._sink = sink
        self._debouncer = KeyDebouncer()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._failure: Optional[str] = None

    def _on_press(self, key) -> None:
        name = _pynput_key_name(key)
        if name is None:
            return
        event = self._debouncer.press(name)
        if event is not None:
            self._sink.put(event)

    def _on_release(self, key) -> None:
        name = _pynput_key_name(key)
        if name is not None:
            self._debouncer.release(name)

    def start(self) -> None:
        self._listener.start()

    def healthy(self) -> bool:
        return bool(self._listener.running)

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:  # best-effort teardown
            pass


def create_keyboard_backend(sink: 'queue.Queue', preferred: str = 'pynput'):
    """Build the best available key backend, or raise with the reason.

    Never silently degrades to a backend that cannot see a standalone Alt
    press: an operator who believes Alt works must actually have it.
    """
    if preferred not in ('pynput', 'none'):
        raise ValueError(f'unknown keyboard backend {preferred!r}')
    if preferred == 'none':
        return None
    if not os.environ.get('DISPLAY'):
        raise RuntimeError(
            'no DISPLAY is set; the pynput X11 backend cannot receive key '
            'events. Launch from a graphical session, or export DISPLAY=:0')
    backend = PynputKeyboardBackend(sink)
    backend.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not backend.healthy():
        time.sleep(0.05)
    if not backend.healthy():
        backend.stop()
        raise RuntimeError('the pynput keyboard listener failed to start')
    return backend


# ── ROS shell ─────────────────────────────────────────────────────────────


def _duration(seconds: float):
    from builtin_interfaces.msg import Duration as DurationMsg

    whole = int(seconds)
    return DurationMsg(sec=whole,
                       nanosec=int(round((seconds - whole) * 1.0e9)))


class RealOperatorControl(Node):
    """Keyboard acquisition, operator interlock, and the authorization beat."""

    def __init__(self) -> None:
        super().__init__('real_operator_control')
        declare = self.declare_parameter
        self.robot_name = str(
            declare('robot_name', 'crazyflie').value).strip('/')
        self.hardware_prefix = str(declare(
            'hardware_service_prefix',
            f'/{self.robot_name}').value).rstrip('/')
        status_topic = str(declare(
            'status_topic', f'/{self.robot_name}/status').value)
        odom_topic = str(declare(
            'odom_topic', f'/{self.robot_name}/odom').value)
        self.authorization_topic = str(declare(
            'authorization_topic', '/real_operator/autonomy_authorized').value)
        state_topic = str(declare(
            'state_topic', '/real_operator/state').value)
        #: The control adapter's virtual Land service.  Using it keeps the
        #: notify_setpoints_stop -> land handover in one place instead of
        #: giving this node a second, competing landing implementation.
        self.adapter_land_service = str(declare(
            'adapter_land_service', '/real_control/land_request').value)
        self.keyboard_backend_name = str(
            declare('keyboard_backend', 'pynput').value)
        self.service_timeout = float(declare('service_timeout_sec', 2.0).value)
        self.land_height = float(declare('land_height_m', 0.05).value)
        self.land_duration = float(declare('land_duration_sec', 3.0).value)
        publish_period = float(declare('publish_period_sec', 0.05).value)

        self.supervisor = OperatorSupervisor(OperatorConfig(
            status_timeout_sec=float(
                declare('status_timeout_sec', 1.50).value),
            odom_timeout_sec=float(declare('odom_timeout_sec', 0.50).value),
            arm_confirmation_timeout_sec=float(
                declare('arm_confirmation_timeout_sec', 3.0).value),
            disarm_confirmation_timeout_sec=float(
                declare('disarm_confirmation_timeout_sec', 3.0).value),
            land_confirmation_timeout_sec=float(
                declare('land_confirmation_timeout_sec', 30.0).value),
            grounded_height_m=float(declare('grounded_height_m', 0.12).value),
            max_start_offset_m=float(
                declare('max_start_offset_m', 5.0).value),
        ))

        live_qos = QoSProfile(depth=1)
        live_qos.history = HistoryPolicy.KEEP_LAST
        live_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos = QoSProfile(depth=1)
        latched_qos.history = HistoryPolicy.KEEP_LAST
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(Status, status_topic, self._on_status,
                                 live_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, live_qos)
        # Transient-local so a late-joining algorithm still sees the gate,
        # but consumers must additionally require freshness: a stopped
        # heartbeat has to revoke authorization, not preserve it.
        self.authorization_pub = self.create_publisher(
            Bool, self.authorization_topic, latched_qos)
        self.state_pub = self.create_publisher(String, state_topic,
                                               latched_qos)

        self._service_clients = {
            'arm': self.create_client(Arm, f'{self.hardware_prefix}/arm'),
            'emergency': self.create_client(
                Empty, f'{self.hardware_prefix}/emergency'),
            'land': self.create_client(Land, self.adapter_land_service),
        }
        self._service_clients['disarm'] = self._service_clients['arm']

        self._events: 'queue.Queue' = queue.Queue()
        self._keyboard = None
        self._keyboard_error: Optional[str] = None
        self._operation: Optional[str] = None
        self._operation_started = 0.0
        self._future = None
        self._last_state = self.supervisor.state
        self._last_authorized: Optional[bool] = None
        self._lock = threading.Lock()

        try:
            self._keyboard = create_keyboard_backend(
                self._events, self.keyboard_backend_name)
        except Exception as error:
            self._keyboard_error = str(error)

        for line in CONTROL_LEGEND.splitlines():
            self.get_logger().info(line)
        if self._keyboard is None and self.keyboard_backend_name != 'none':
            self.get_logger().error(
                f'KEYBOARD BACKEND UNAVAILABLE: {self._keyboard_error}. '
                'Autonomy cannot be authorized and the aircraft cannot be '
                'armed from this node.')
        else:
            self.get_logger().info(
                f'operator keyboard backend: '
                f'{self.keyboard_backend_name}; robot {self.robot_name}')
        self.get_logger().info(
            f'OPERATOR STATE: {self.supervisor.state.value}')

        self.create_timer(max(0.01, publish_period), self._tick)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    # ── telemetry callbacks ───────────────────────────────────────────────

    def _on_status(self, msg: Status) -> None:
        with self._lock:
            self.supervisor.update_status(msg.supervisor_info, self._now())

    def _on_odom(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        with self._lock:
            self.supervisor.update_odometry(
                position.z, self._now(), position.x, position.y)

    # ── service driving ───────────────────────────────────────────────────

    def _request_for(self, operation: str):
        if operation == 'arm':
            return Arm.Request(arm=True)
        if operation == 'disarm':
            return Arm.Request(arm=False)
        if operation == 'emergency':
            return Empty.Request()
        if operation == 'land':
            return Land.Request(group_mask=0, height=self.land_height,
                                duration=_duration(self.land_duration))
        raise ValueError(f'unknown operator operation {operation}')

    def _service_finished(self, operation: str, future) -> None:
        success = False
        try:
            success = future.result() is not None
        except Exception as error:
            self.get_logger().error(f'{operation} service failed: {error}')
        self._operation = None
        self._future = None
        with self._lock:
            message = self.supervisor.service_result(
                operation, success, self._now())
        if message:
            level = self.get_logger().info if success \
                else self.get_logger().error
            level(message)

    def _drive_service(self, now: float) -> None:
        with self._lock:
            required = self.supervisor.required_service()
        if required is None:
            self._operation = None
            self._future = None
            return
        if self._operation != required:
            self._operation = required
            self._operation_started = now
            self._future = None
        client = self._service_clients[required]
        if self._future is None and client.service_is_ready():
            self._future = client.call_async(self._request_for(required))
            self._future.add_done_callback(
                lambda future, name=required:
                self._service_finished(name, future))
            return
        if now - self._operation_started > self.service_timeout:
            self.get_logger().error(
                f'{required} service unavailable or timed out')
            self._operation = None
            self._future = None
            with self._lock:
                self.supervisor.service_result(required, False, now)

    # ── main loop ─────────────────────────────────────────────────────────

    def _drain_events(self, now: float) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                message = self.supervisor.on_key(event, now)
            if not message:
                continue
            if 'REJECTED' in message or 'FAILED' in message \
                    or 'EMERGENCY' in message:
                self.get_logger().error(message)
            else:
                self.get_logger().info(message)

    def _tick(self) -> None:
        now = self._now()
        if (self._keyboard is not None and not self._keyboard.healthy()
                and self._keyboard_error is None):
            self._keyboard_error = 'keyboard listener stopped unexpectedly'
            self.get_logger().error(
                'KEYBOARD LISTENER LOST - operator control is degraded; '
                'autonomy authorization will be withheld')
        self._drain_events(now)
        with self._lock:
            if self._keyboard_error is not None:
                # Fail closed: without a working abort key the gate stays shut.
                self.supervisor.land_latched = True
            decision = self.supervisor.tick(now)
        if decision.message:
            self.get_logger().warning(decision.message)
        if decision.state != self._last_state:
            self.get_logger().info(
                f'OPERATOR STATE: {decision.state.value}')
            self.state_pub.publish(String(data=decision.state.value))
            self._last_state = decision.state
        if decision.authorized != self._last_authorized:
            self.get_logger().warning(
                f'AUTONOMY AUTHORIZED: {decision.authorized}')
            self._last_authorized = decision.authorized
        self.authorization_pub.publish(Bool(data=decision.authorized))
        self._drive_service(now)

    def destroy_node(self) -> bool:
        if self._keyboard is not None:
            self._keyboard.stop()
        return super().destroy_node()


def main(args=None) -> None:
    if not ROS_AVAILABLE:
        raise RuntimeError('ROS 2 Python and crazyflie_interfaces are required')
    rclpy.init(args=args)
    node = RealOperatorControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
