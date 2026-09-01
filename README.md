# Crazyflie Autonomous Exploration, Mapping and Navigation

Autonomous layer-by-layer exploration and saved-map waypoint navigation for a
Crazyflie 2.1 with a Flow deck and a Multi-Ranger deck, built on ROS 2 Humble,
Gazebo (via `ros_gz`) and RViz2. The aircraft carries six single-ray ToF
sensors and no camera or 360° lidar, so the world is mapped as a **stack of 2D
occupancy grids, one per altitude layer** ("2.5D") rather than as a single
plane or a full 3D volume.

> **Experimental research software**, written for a bachelor thesis. The
> complete workflow — explore → save layer maps → fly a multi-layer waypoint
> mission → land — is validated **in simulation**. Physical Crazyflie testing
> covered a smaller subset (see [Real Crazyflie](#real-crazyflie)). Nothing
> here is production software.

## Overview

Two independent algorithms, both in the ROS 2 package
[`ros2_ws/src/cf_explore`](ros2_ws/src/cf_explore). They never run at the same
time — only one node may publish `/cmd_vel`.

- **`layer_explore`** — flies an unknown world autonomously, builds one
  occupancy grid per altitude layer, and saves each layer to disk as
  PGM + YAML + JSON.
- **`cf_auto`** — flies a configured 3D waypoint mission over those *saved*
  maps: AMCL localization, A\* planning, pure-pursuit following, layer changes,
  and a ranger-based landing.

Each algorithm has a simulation launch and a real-hardware launch, giving four
supported workflows:

| Workflow | Launch | What it does |
|---|---|---|
| Simulation mapping | `layer_explore.launch.py` | explore and save layer maps in Gazebo |
| Simulation navigation | `cf_auto.launch.py` | fly a waypoint mission over saved maps in Gazebo |
| Real mapping | `layer_explore_real.launch.py` | the same explorer on the physical aircraft |
| Real navigation | `cf_auto_real.launch.py` | the same navigator on the physical aircraft |

The two real launches share [`real_base.launch.py`](ros2_ws/src/cf_explore/launch/real_base.launch.py),
which owns the hardware boundary and its safety gates (see
[Real Crazyflie](#real-crazyflie)).

## Features

- Frontier exploration with a 120° yaw sweep per scan, connected-component
  reachability filtering and cluster scoring.
- Per-altitude occupancy mapping; layer altitudes are **measured in flight**
  from the floor/ceiling planes, not hard-coded.
- 8-connected A\* that treats unknown cells as untraversable, with obstacle
  inflation and line-of-sight path simplification.
- Static multi-layer 3D routing over the whole saved map stack — the route is
  discrete in `(cell x, cell y, layer)`, and its cost is the metres actually
  flown (horizontal + vertical) with no artificial layer-change penalty.
- **Diagonal inter-layer transition execution**: a layer hop can move in XY and
  Z simultaneously instead of climbing in place, after its corridor is
  validated against both adjacent inflated maps; otherwise it falls back to the
  in-place climb (see [Layer transitions](#layer-transitions)).
- Runtime map switching with automatic AMCL reseeding — no second RViz click.
- Live collision guard on a separate, unfiltered scan topic, plus a last-resort
  vertical bypass for obstacles that are physically present but in no saved map.
- Landing driven by the down-facing ranger, which refuses to descend on stale
  data.
- Real-hardware boundary (adapters, safety watchdog, operator keyboard gate)
  kept strictly separate from the simulation path.

## Tested environment

| Component | Version used |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| ROS 2 | Humble Hawksbill |
| Gazebo | Harmonic (`gz-sim` 8.15.0), via `ros-humble-ros-gzharmonic` |
| Python | 3.10.12 |
| numpy / scipy / PyYAML | 2.2.6 / 1.15.3 / 6.0.3 |
| Crazyswarm2 | 1.0.5 (vendored, see below) |

Versions are what this project was developed and tested against, not pinned
requirements.

## Repository structure

```
ros2_ws/
├── map/                       # saved simulation layer maps (the maps cf_auto flies)
│   └── original_map/          # unedited layer_explore output, kept for reference
├── map_real/                  # real-hardware maps (currently empty)
└── src/
    ├── cf_explore/            # the project package
    │   ├── cf_explore/        # nodes and pure algorithm libraries
    │   ├── launch/
    │   ├── config/            # node params + RViz configs
    │   ├── worlds/            # test-only Gazebo worlds
    │   └── test/              # pytest suite
    ├── crazyswarm2/                 # vendored external packages, do not edit
    ├── crazyflie_ros2_multiranger/
    └── ros_gz_crazyflie/            # Gazebo world, model, bridges, control
simulation_ws/                 # external asset checkout, not used by ros2_ws
prototype/                     # unrelated C scratch prototype, not part of the ROS 2 project
```

The three packages under `ros2_ws/src` other than `cf_explore` are **vendored
copies** (checked in directly, not git submodules) and carry project-specific
modifications — the Gazebo world and Crazyflie model, the `ros_gz` bridge
config, the mapping RViz config and the Crazyswarm2 inventory. Do not edit them
as part of normal work on this project.

## Requirements

- Ubuntu 22.04 with ROS 2 Humble (`ros-humble-desktop`)
- Gazebo Harmonic and `ros_gz` (`ros-humble-ros-gzharmonic`)
- `ros-humble-nav2-map-server`, `ros-humble-nav2-amcl`,
  `ros-humble-nav2-lifecycle-manager` (`cf_auto` only)
- `python3-numpy`, `python3-scipy`, `python3-yaml`
- `python3-pynput` — only for the real-hardware operator keyboard. It is
  **not** declared in [`package.xml`](ros2_ws/src/cf_explore/package.xml), so
  `rosdep` will not install it; install it manually if you fly hardware.

Install the declared dependencies:

```bash
cd ~/crazyflie_autonomous_exploration_mapping_navigation/ros2_ws
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

`rosdep` prints `Cannot locate rosdep definition for [ament_python]` for
`cf_explore` — `ament_python` is a build type, not a rosdep key. `-r` makes
rosdep continue and install everything else, so this message is expected and
harmless. `rosdep` does not cover Gazebo itself or `pynput`; install those from
the distribution packages listed above.

## Build

First build (builds the vendored packages too):

```bash
cd ~/crazyflie_autonomous_exploration_mapping_navigation/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

Rebuild only this project's package:

```bash
cd ~/crazyflie_autonomous_exploration_mapping_navigation/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select cf_explore --symlink-install
source install/setup.bash
```

Re-source an already built workspace in a new terminal:

```bash
source /opt/ros/humble/setup.bash && source ~/crazyflie_autonomous_exploration_mapping_navigation/ros2_ws/install/setup.bash
```

## Run the mapping simulation

One command starts Gazebo, the bridges, the TF tree, `layer_explore` and RViz2:

```bash
ros2 launch cf_explore layer_explore.launch.py map_save_dir:=/tmp/cf_explore_maps
```

Takeoff is automatic; there is nothing to click. The drone probes the room
height, then repeats `SCAN → SELECT → NAVIGATE` until no reachable frontier
remains, saves that layer, climbs to the next one, and finally lands.

> **`map_save_dir` matters.** Left at its default the run writes into
> [`ros2_ws/map/`](ros2_ws/map) and **overwrites the saved navigation maps**

Useful arguments: `rviz:=False`, `scan_rotation_angle_deg:=120.0`,
`scan_yaw_rate:=0.40`, `cruise_speed_mps:=0.80`.

## Run autonomous navigation

```bash
ros2 launch cf_explore cf_auto.launch.py
```

This starts Gazebo, both scan mergers, `nav2_map_server` + `nav2_amcl` +
lifecycle manager, the layer visualizer, `cf_auto` and RViz2.

**The mission does not start on its own.** `cf_auto` boots into
`WAIT_FOR_INITIAL_POSE` and stays motionless until it has a `/map`, a pose on
`/initialpose`, and a valid `map → base` transform. After RViz opens, click
**"2D Pose Estimate"** and drag at the drone's actual starting pose and
heading. Takeoff follows immediately.

Useful arguments: `rviz:=False`, `layer_markers:=False`, `params_file:=<yaml>`,
`map_dir:=<dir>`, or an explicit stack with
`map_yaml:=<dir>/map_layer_1.yaml extra_layer_maps:=<dir>/map_layer_2.yaml,...`.
`world:=<absolute .sdf path>` starts the same stack on a different Gazebo
world; left empty it uses the bundled one.

## Maps and configuration

Saved maps live in [`ros2_ws/map/`](ros2_ws/map), three files per layer:

| File | Contents |
|---|---|
| `map_layer_N.pgm` | occupancy image (occupied 0, free 254, unknown 205) |
| `map_layer_N.yaml` | `resolution`, `origin`, thresholds — read by `nav2_map_server` |
| `map_layer_N.json` | `z_height`, the authoritative altitude of that layer |

A layer counts only when all three files exist, and layer numbering must be
contiguous from 1. **Nothing writes the layer count down twice**:
`cf_auto.launch.py` discovers the stack from the map directory and reads each
altitude from that layer's own `.json`, so adding or removing a saved layer
needs no edit to any config file. The number of layers, their altitudes and the
grid geometry are properties of whatever map set you point the launch at, not
of the algorithm.

> **Example dataset.** The map set committed in `ros2_ws/map/` happens to hold
> three layers at 0.50 / 1.00 / 1.50 m, 650×650 cells at 0.05 m, origin
> `(-16.25, -16.25)`. These are the values of this one saved experiment; a
> different exploration run produces a different stack and needs no code change.

`ros2_ws/map/original_map/` holds the unedited `layer_explore` output; the maps
in `ros2_ws/map/` have been cleaned up by hand since. `ros2_ws/map_real/` is
the real-hardware map directory and is currently empty.

At launch, `cf_auto.launch.py` writes corrected copies of each map YAML into
`$TMPDIR/cf_auto/` with `free_thresh` clamped to 0.196, so that unknown pixels
(205) stay unknown instead of silently becoming free space. **The saved map
files are never modified.**

Configuration lives in
[`ros2_ws/src/cf_explore/config/`](ros2_ws/src/cf_explore/config):

| File | Used by |
|---|---|
| `cf_auto.yaml` | `cf_auto`, `map_server`, `amcl`, both scan mergers, visualizer |
| `cf_auto.rviz` | the navigation RViz view — shared by `cf_auto` **and** `cf_auto_real` |
| `layer_explore_real.yaml`, `layer_explore_real.rviz` | real mapping |
| `cf_auto_real.yaml` | real navigation |
| `real_safety.yaml`, `crazyflies_real.yaml` | shared by both real workflows |

`layer_explore` has no simulation params file — its defaults are in
[`layer_explore.py`](ros2_ws/src/cf_explore/cf_explore/layer_explore.py) and
are overridden from the launch file.

## Layer transitions

Multi-layer routing is **discrete**: `layer_route.plan_3d_route` searches
`(cell x, cell y, layer)` over the saved grids and emits `MOVE` legs and
`TRANSITION` legs. A transition cell must be free on **both** adjacent
inflated grids. This is not free-space 3D trajectory planning.

**Where a hop happens is derived, not configured.** The static planner finds
the transition cell in the saved maps, so a mission needs no hand-measured
coordinates — in particular a real mission never requires them. The
`transition_*` parameters are a **fallback** table, read only where the static
planner is unavailable (multi-layer routing off, or a layer with no cached
grid). While the static planner owns transitions, that table is unused, and
preflight validation skips it rather than letting an unused placeholder abort
a valid mission; the moment the fallback can actually be reached, its strict
known-free and inflation checks and its missing-hop abort both apply again.

How a `TRANSITION` leg is *flown* is a separate, execution-level decision:

1. **Diagonal (default).** If the next leg is a `MOVE` on the target layer,
   `plan_diagonal_endpoint` walks along that already-planned polyline to pick
   an endpoint B. The reach is derived from the configured speeds
   (`transition_xy_speed_mps` × altitude change ÷ `ascend_speed`/`descend_speed`),
   never guessed.
   The whole A→B segment is sampled every 0.05 m and every sample must be free
   on **both** the source and target inflated grids; the span is shortened
   until it clears. During the climb the XY hold target slides along A→B in
   step with altitude progress, still passing through the live collision guard.
2. **Vertical fallback.** Feature disabled, no target-layer path, or no safe
   corridor → the validated in-place climb, unchanged.

Either way the map switch happens **after** the target altitude is reached
(`/map_server/load_map`, waiting for the republished `/map`), and AMCL is then
reseeded at the pose the aircraft actually reached — near B for a diagonal hop,
at A for a vertical one.

Parameters (both are code defaults, not present in `cf_auto.yaml`):
`diagonal_layer_transitions_enabled` (default `true`) and
`transition_xy_speed_mps` (default `0.50`).

## Run tests

```bash
cd ~/crazyflie_autonomous_exploration_mapping_navigation/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
cd src/cf_explore && python3 -m pytest test/
```

Source the ROS environment **first** — without it every test module fails at
collection with `ModuleNotFoundError: rclpy`, which looks like a broken tree
and is not.

> Use `pytest`, not `colcon test`. `colcon test` collects only the
> `unittest.TestCase` classes in `test_colcon_geometry.py` — 33 of the roughly
> 980 tests — and reports success without running the rest.

The suite is pure unit tests: no ROS graph, no Gazebo, no hardware.

## Real Crazyflie

**Hardware:** Crazyflie 2.1 Brushless with a Flow deck and a Multi-Ranger
deck, flown over a Crazyradio USB dongle. The Crazyswarm2 inventory
[`config/crazyflies_real.yaml`](ros2_ws/src/cf_explore/config/crazyflies_real.yaml)
ships with **placeholder identity** (`__ROBOT_NAME__`, `__RADIO_URI__`,
`enabled: false`); the real name and radio URI are supplied at launch and are
deliberately not stored in the repository.

The real path is a separate set of launch files and nodes, and is
**fail-closed by default**: `real_base.launch.py` starts with the robot
disabled, `dry_run:=true`, `autonomy_enabled:=false` and
`extrinsics_verified:=false`. Enabling hardware requires explicit identity and
extrinsics attestations as launch arguments, and even then:

- autonomy can **never** arm the aircraft — arming is an operator keypress
  only, enforced independently in `real_operator_control` and
  `real_control_adapter`;
- every velocity passes a safety watchdog permit and the control adapter's
  freshness gates;
- the operator keyboard (`Left Alt` arm/disarm, `G` authorize autonomy,
  `L` land, `SPACE` emergency motor cut) is the only path from a keypress to
  motion. `pynput` installs a **global** X11 hook: while the stack runs it
  captures keys from any window, so do not type anything unrelated while the
  aircraft is armed.

Because those gates need physical attestations, a real flight is not a single
copy-pasteable command and none is given here. Inspect
[`launch/real_base.launch.py`](ros2_ws/src/cf_explore/launch/real_base.launch.py),
[`launch/layer_explore_real.launch.py`](ros2_ws/src/cf_explore/launch/layer_explore_real.launch.py),
[`launch/cf_auto_real.launch.py`](ros2_ws/src/cf_explore/launch/cf_auto_real.launch.py)
and [`config/real_safety.yaml`](ros2_ws/src/cf_explore/config/real_safety.yaml)
before attempting one, and see `--show-args` for the full gate list of either
real workflow:

```bash
ros2 launch cf_explore layer_explore_real.launch.py --show-args

ros2 launch cf_explore layer_explore_real.launch.py \
  robot_name:=crazyflie \
  radio_uri:=radio://0/80/2M/E7E7E7E7E7 \
  hardware_identity_confirmed:=true \
  autonomy_enabled:=true \
  dry_run:=false \
  extrinsics_verified:=true \
  front_xyz:=0,0,0.02 \
  right_xyz:=0,0,0.02 \
  back_xyz:=0,0,0.02 \
  left_xyz:=0,0,0.02 \
  up_xyz:=0,0,0.02 \
  down_xyz:=0,0,0.02 \
  front_rpy:=0,0,0 \
  right_rpy:=0,0,-1.57079632679 \
  back_rpy:=0,0,3.14159265359 \
  left_rpy:=0,0,1.57079632679 \
  up_rpy:=0,-1.57079632679,0 \
  down_rpy:=0,1.57079632679,0
```

```bash
ros2 launch cf_explore cf_auto_real.launch.py --show-args

ros2 launch cf_explore cf_auto_real.launch.py \
  robot_name:=crazyflie \
  radio_uri:=radio://0/80/2M/E7E7E7E7E7 \
  hardware_identity_confirmed:=true \
  autonomy_enabled:=true \
  dry_run:=false \
  extrinsics_verified:=true \
  mission_waypoints_xyz:="x1,y1,z1,x2,y2,z2" \
  front_xyz:=0,0,0.02 \
  right_xyz:=0,0,0.02 \
  back_xyz:=0,0,0.02 \
  left_xyz:=0,0,0.02 \
  up_xyz:=0,0,0.02 \
  down_xyz:=0,0,0.02 \
  front_rpy:=0,0,0 \
  right_rpy:=0,0,-1.57079632679 \
  back_rpy:=0,0,3.14159265359 \
  left_rpy:=0,0,1.57079632679 \
  up_rpy:=0,-1.57079632679,0 \
  down_rpy:=0,1.57079632679,0
```

That single command brings up the whole real navigation stack: the hardware
boundary from `real_base.launch.py` (sensor adapter, safety watchdog, control
adapter, operator keyboard, body frame), `nav2_map_server`, `nav2_amcl` and
its lifecycle manager, both scan mergers, the planar frame, `cf_auto`, the
layer visualizer and **RViz2** — no second terminal.

`cf_auto_real` uses the *same* [`config/cf_auto.rviz`](ros2_ws/src/cf_explore/config/cf_auto.rviz)
view as the simulation, because every display in it (`/map`, `/scan`,
`/cf_auto/path`, `/cf_auto/waypoints`, `/layer_map_markers`, `/amcl_pose`) is
published identically by the real stack and its fixed frame is `map` either
way. As in simulation, **the mission does not start on its own**: `cf_auto`
holds in `WAIT_FOR_INITIAL_POSE` until it has a `/map`, an `/initialpose` and
a valid `map → base` transform, so click **"2D Pose Estimate"** in RViz and
drag at the aircraft's actual starting pose and heading.

RViz is a viewer, not an authorization: it opens before the aircraft is armed,
it publishes no command and owns no safety state, and arming, autonomy, landing
and the emergency cut remain `Left Alt`, `G`, `L` and `SPACE` on the operator
keyboard. Add `rviz:=False` for an explicit headless run, and
`layer_markers:=False` to omit the saved-layer markers.

### Mission scope of the real configuration

The shipped real profiles impose **no artificial mission bound**. Once the
operator has armed with `Left Alt` and authorized autonomy with `G`:

- `layer_explore_real` runs the complete exploration mission — `TAKEOFF →
  PROBE → SCAN → SELECT → NAVIGATE →` repeat `→` layer complete `→` save `→`
  next layer `→ LAND → DONE` — and stops on its own completion criteria. The
  layer count is derived in flight from the measured floor and ceiling; it is
  not written down anywhere.
- `cf_auto_real` runs the complete waypoint mission — localization, takeoff,
  planning, path following, layer transitions, map switching, relocalization,
  every configured waypoint, then landing — with multi-layer routing, the
  vertical bypass and the generic replan budget all active.

`max_layers`, `halt_after_state` and `halt_after_layer` remain available as
generic optional debugging bounds, but **none of them is set** in
[`layer_explore_real.yaml`](ros2_ws/src/cf_explore/config/layer_explore_real.yaml)
or [`cf_auto_real.yaml`](ros2_ws/src/cf_explore/config/cf_auto_real.yaml).
Only the internal fail-safes remain: stale-data rejection, the motion permit,
the collision guard, controlled landing and post-landing disarm. None of them
needs operator interaction during a healthy flight.

### What has actually been flown

**Software capability is not flight evidence.** The physical testing described
below predates the configuration above and was deliberately bounded; nothing
here has been re-flown since the mission bounds were removed.

Earlier physical testing used `layer_explore` only, on a single layer, in
**bounded supervised runs**: that configuration set `halt_after_state` so an
experiment ended at a chosen state before the operator had to react. Flight
logs are kept locally under `ros2_ws/log/real_flight_*` and are not committed.

Demonstrated in the air at that time, with logged evidence: radio link and
telemetry, the Multi-Ranger and TF pipeline, the operator keyboard (arm /
authorize / land / emergency), takeoff, the room-height probe, the 120° yaw
scan, live occupancy updates, the bounded validation hold, and a controlled
landing followed by disarm — i.e. the chain `TAKEOFF → PROBE → SCAN →
VALIDATION_HOLD → land → disarmed`. Later supervised runs also exercised
frontier selection, A\* routing and short navigation legs; treat any claim
beyond the chain above as provisional unless you have the corresponding log.

**Never executed on the physical aircraft:** completing a layer, saving a real
map (`ros2_ws/map_real/` is empty), the climb to a second layer, multi-layer
real mapping, and the whole of `cf_auto` — real navigation, real map switching
and real layer transitions, diagonal or vertical. All of these are implemented
and, since the mission bounds were removed, also **enabled** in the real
configuration — but they remain **unvalidated on hardware**. Enabled is not
flown.

## Known limitations

- **Experimental research software.** Parameter values are simulation starting
  points; several are explicitly marked in the configs as unmeasured.
- **Altitude is discrete.** The world is represented as separate mapped layers,
  not as a continuous 3D volume. `cf_auto` waypoints must sit on a saved layer
  altitude within `layer_altitude_tolerance_m`, or the mission aborts before
  takeoff.
- **`cf_auto` flies static maps.** Only the collision guard and the vertical
  bypass react to obstacles that are not in the saved maps.
- **Manual initial pose.** `cf_auto` needs an RViz "2D Pose Estimate" to start;
  there is no automatic global localization.
- **Layer altitude follows the terrain.** The firmware's altitude estimate is
  derived from the down-facing ToF sensor, so flying over a raised surface
  raises the aircraft with it and a mapping layer is not a perfectly fixed
  plane in the room. The mechanism is documented in
  [`layer_altitude.py`](ros2_ws/src/cf_explore/cf_explore/layer_altitude.py); a
  compensation prototype lives there but is **off by default** and unflown.
- **Early exploration can route through unobserved space.** A\* rejects unknown
  cells and inflates around *known* obstacles, so after only a couple of scans
  a route can cross space whose obstacles have not been observed yet. On
  hardware this ended one exploration leg in a collision. It is the main open
  algorithmic limitation for multi-frontier exploration.
- **Physical coverage is far narrower than simulation coverage** — see
  [Real Crazyflie](#real-crazyflie).
- Simulation transition points in `cf_auto.yaml` are hand-measured for the
  bundled world, but they are only the fallback table (see
  [Layer transitions](#layer-transitions)); a hop naming a layer the map
  directory does not have is dropped at launch.

## License and author

`cf_explore` is declared MIT in
[`package.xml`](ros2_ws/src/cf_explore/package.xml). The vendored packages
under `ros2_ws/src` keep their own upstream licenses (see the `LICENSE` file in
each). Bachelor thesis project by **fabzak**.
