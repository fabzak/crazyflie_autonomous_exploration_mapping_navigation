/* Architecture-B terrain-rejection datum lifecycle - standalone host C.
 *
 * PROTOTYPE.  Not part of any Crazyflie firmware build.  No ROS, no CRTP,
 * no FreeRTOS, no dynamic allocation, no mutable globals.
 *
 * Behavioural oracle: oracle_datum_lifecycle.py
 *   md5 c453b7427117b27b3ff631e029248a32
 * This module must match that oracle's semantics exactly.
 *
 * The datum is an ATTITUDE-CORRECTED FLOOR CLEARANCE (down_range * R22),
 * never odom.z / stateEstimate.z: the estimator's Z is the terrain-
 * contaminated quantity that created the problem in the first place.
 */
#ifndef DATUM_LIFECYCLE_H
#define DATUM_LIFECYCLE_H

#include <stdbool.h>
#include <stdint.h>

/* ---- frozen thresholds (validated in simulation; do not tune here) ---- */
#define DL_TERRAIN_THRESH_M      0.015f
#define DL_REVALIDATE_TOL_M      0.030f
#define DL_STABLE_TOL_M          0.010f
#define DL_N_STABLE              20u
#define DL_REVALIDATE_DWELL_MS   1000u
#define DL_SETTLE_DWELL_MS        500u
#define DL_VZ_CMD_EPS            0.02f
#define DL_VZ_EST_EPS            0.05f
#define DL_XY_EPS                0.03f
#define DL_TILT_MAX_DEG          10.0f
#define DL_PHASE_TIMEOUT_MS       500u
#define DL_REVALIDATE_TIMEOUT_MS 5000u
#define DL_TRANSITION_TIMEOUT_MS 15000u
#define DL_ACQUIRE_TIMEOUT_MS    8000u
#define DL_ABORT_TIMEOUT_MS      3000u

typedef enum {
    DL_PHASE_UNKNOWN = 0,
    DL_PHASE_HOLD_LAYER,
    DL_PHASE_VERTICAL_TRANSITION,
    DL_PHASE_LANDING
} dl_phase_t;

typedef enum {
    DL_UNINIT = 0,
    DL_DATUM_ACQUIRE,
    DL_DATUM_VALID,
    DL_VERTICAL_TRANSITION,
    DL_SETTLING,
    DL_DATUM_REVALIDATE,
    DL_TERRAIN_SUSPECT,
    DL_BRAKE,
    DL_RETREAT,
    DL_RECOVER,
    DL_ABORT
} dl_state_t;

typedef struct {
    uint32_t now_ms;         /* monotonic; wraparound-safe differences */
    bool     clearance_valid;/* false == the oracle's `clearance is None` */
    float    clearance_m;    /* down_range * R22, metres */
    dl_phase_t phase;
    uint32_t phase_age_ms;
    float    vz_cmd_mps;
    float    vz_est_mps;
    float    vxy_mps;
    float    tilt_deg;
    bool     delta_z_known;  /* false == the oracle's `delta_z is None` */
    float    delta_z_m;      /* from the LAYER DEFINITION, not integrated vz */
    bool     request_transition;
} dl_input_t;

typedef struct {
    bool       reject_tof_update;   /* SAFETY: do not let this sample reach
                                     * the Kalman scalar update */
    bool       datum_valid;
    bool       detector_enabled;
    bool       transition_active;
    bool       tof_fuse_allowed;
    bool       xy_allowed;
    bool       z_cmd_allowed;
    bool       recovery_required;
    bool       abort_required;
    dl_state_t state;
    float      datum_clearance_m;   /* NAN when none */
    float      expected_clearance_m;/* NAN when none */
    uint32_t   datum_epoch;
} dl_output_t;

typedef struct {
    dl_state_t state;
    bool     datum_set;
    float    datum_clearance_m;
    bool     expected_set;
    float    expected_clearance_m;
    bool     old_datum_set;
    float    old_datum_clearance_m;
    uint32_t datum_epoch;
    uint32_t t_state_ms;
    bool     t_stable_set;
    uint32_t t_stable_ms;
    float    window[DL_N_STABLE];
    uint32_t window_len;
    uint32_t rejected_count;
    uint32_t transition_rejected_count;
} dl_t;

void dl_init(dl_t *dl);
void dl_update(dl_t *dl, const dl_input_t *in, dl_output_t *out);
const char *dl_state_name(dl_state_t s);

#endif /* DATUM_LIFECYCLE_H */
