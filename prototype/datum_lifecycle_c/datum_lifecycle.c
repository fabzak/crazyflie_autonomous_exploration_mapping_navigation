/* Architecture-B datum lifecycle - standalone host C implementation.
 * Semantics mirror oracle_datum_lifecycle.py (md5 c453b742...) exactly. */

#include "datum_lifecycle.h"
#include <math.h>
#include <string.h>

/* Wraparound-safe elapsed time: valid for any uint32_t monotonic counter. */
static inline uint32_t dl_elapsed(uint32_t now, uint32_t since)
{
    return (uint32_t)(now - since);
}

static inline bool dl_finite(float v)
{
    return isfinite(v) != 0;
}

static void dl_window_clear(dl_t *dl)
{
    dl->window_len = 0u;
}

/* Append and report stability, mirroring the oracle's _stable(): the window
 * holds the last DL_N_STABLE samples; stable == full window with
 * peak-to-peak <= DL_STABLE_TOL_M. */
static bool dl_window_push_stable(dl_t *dl, float clearance)
{
    if (dl->window_len < DL_N_STABLE) {
        dl->window[dl->window_len++] = clearance;
    } else {
        for (uint32_t i = 1u; i < DL_N_STABLE; i++) {
            dl->window[i - 1u] = dl->window[i];
        }
        dl->window[DL_N_STABLE - 1u] = clearance;
    }
    if (dl->window_len < DL_N_STABLE) {
        return false;
    }
    float lo = dl->window[0], hi = dl->window[0];
    for (uint32_t i = 1u; i < DL_N_STABLE; i++) {
        if (dl->window[i] < lo) lo = dl->window[i];
        if (dl->window[i] > hi) hi = dl->window[i];
    }
    return (hi - lo) <= DL_STABLE_TOL_M;
}

/* sorted(window)[len//2] - the oracle's upper median. */
static float dl_window_median(const dl_t *dl)
{
    float tmp[DL_N_STABLE];
    uint32_t n = dl->window_len;
    if (n == 0u) return NAN;
    memcpy(tmp, dl->window, (size_t)n * sizeof(float));
    for (uint32_t i = 1u; i < n; i++) {           /* insertion sort, no malloc */
        float key = tmp[i];
        uint32_t j = i;
        while (j > 0u && tmp[j - 1u] > key) { tmp[j] = tmp[j - 1u]; j--; }
        tmp[j] = key;
    }
    return tmp[n / 2u];
}

static void dl_go(dl_t *dl, dl_state_t s, uint32_t now_ms)
{
    if (s == dl->state) return;
    dl->state = s;
    dl->t_state_ms = now_ms;
    dl->t_stable_set = false;
    dl_window_clear(dl);
}

void dl_init(dl_t *dl)
{
    memset(dl, 0, sizeof(*dl));
    dl->state = DL_UNINIT;
    dl->datum_clearance_m = NAN;
    dl->expected_clearance_m = NAN;
    dl->old_datum_clearance_m = NAN;
}

const char *dl_state_name(dl_state_t s)
{
    switch (s) {
    case DL_UNINIT:              return "UNINIT";
    case DL_DATUM_ACQUIRE:       return "DATUM_ACQUIRE";
    case DL_DATUM_VALID:         return "DATUM_VALID";
    case DL_VERTICAL_TRANSITION: return "VERTICAL_TRANSITION";
    case DL_SETTLING:            return "SETTLING";
    case DL_DATUM_REVALIDATE:    return "DATUM_REVALIDATE";
    case DL_TERRAIN_SUSPECT:     return "TERRAIN_SUSPECT";
    case DL_BRAKE:               return "BRAKE";
    case DL_RETREAT:             return "RETREAT";
    case DL_RECOVER:             return "RECOVER";
    case DL_ABORT:               return "ABORT";
    default:                     return "?";
    }
}

static bool dl_is_suspect(dl_state_t s)
{
    return s == DL_TERRAIN_SUSPECT || s == DL_BRAKE || s == DL_RETREAT;
}

static void dl_publish(const dl_t *dl, bool reject, dl_output_t *out)
{
    out->reject_tof_update = reject;
    out->state             = dl->state;
    out->detector_enabled  = (dl->state == DL_DATUM_VALID);
    out->datum_valid       = (dl->state == DL_DATUM_VALID) && dl->datum_set;
    out->transition_active = (dl->state == DL_VERTICAL_TRANSITION);
    out->tof_fuse_allowed  = !dl_is_suspect(dl->state);
    out->xy_allowed        = (dl->state == DL_DATUM_VALID);
    out->z_cmd_allowed     = (dl->state == DL_VERTICAL_TRANSITION) ||
                             (dl->state == DL_ABORT);
    out->recovery_required = dl_is_suspect(dl->state) ||
                             (dl->state == DL_RECOVER);
    out->abort_required    = (dl->state == DL_ABORT);
    out->datum_clearance_m    = dl->datum_set ? dl->datum_clearance_m : NAN;
    out->expected_clearance_m = dl->expected_set ? dl->expected_clearance_m
                                                 : NAN;
    out->datum_epoch       = dl->datum_epoch;
}

void dl_update(dl_t *dl, const dl_input_t *in, dl_output_t *out)
{
    /* Defence in depth: the oracle expresses "invalid" as clearance_valid
     * == false and never sees NaN/Inf.  Screen them anyway so a NaN can
     * never compare its way into an accepted measurement or a validated
     * datum.  On the oracle's input domain this changes nothing. */
    bool geom_ok = in->clearance_valid &&
                   dl_finite(in->clearance_m) &&
                   dl_finite(in->tilt_deg) &&
                   (in->tilt_deg <= DL_TILT_MAX_DEG);

    dl_phase_t phase = in->phase;
    if (in->phase_age_ms > DL_PHASE_TIMEOUT_MS) {
        phase = DL_PHASE_UNKNOWN;      /* fail-safe on stale mission intent */
    }

    const uint32_t now = in->now_ms;
    const dl_state_t st = dl->state;

    if (st == DL_ABORT) {              /* terminal except explicit dl_init */
        dl_publish(dl, false, out);
        return;
    }

    if (in->request_transition && dl_is_suspect(st)) {
        dl->transition_rejected_count++;   /* never bypass fail-closed recovery */
    }

    bool reject = false;

    switch (st) {
    case DL_UNINIT:
        if (geom_ok && phase == DL_PHASE_HOLD_LAYER) {
            dl_go(dl, DL_DATUM_ACQUIRE, now);
        }
        break;

    case DL_DATUM_ACQUIRE:
        if (!geom_ok) {
            dl_window_clear(dl);
        } else if (dl_window_push_stable(dl, in->clearance_m)) {
            dl->datum_clearance_m = dl_window_median(dl);
            dl->datum_set = true;
            dl->datum_epoch++;
            dl_go(dl, DL_DATUM_VALID, now);
        } else if (dl_elapsed(now, dl->t_state_ms) > DL_ACQUIRE_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        break;

    case DL_DATUM_VALID:
        if (phase == DL_PHASE_VERTICAL_TRANSITION) {
            dl->old_datum_clearance_m = dl->datum_clearance_m;
            dl->old_datum_set = dl->datum_set;
            if (in->delta_z_known && dl_finite(in->delta_z_m) &&
                dl->datum_set) {
                dl->expected_clearance_m = dl->datum_clearance_m +
                                           in->delta_z_m;
                dl->expected_set = true;
            } else {
                dl->expected_set = false;
                dl->expected_clearance_m = NAN;
            }
            dl_go(dl, DL_VERTICAL_TRANSITION, now);
        } else if (phase == DL_PHASE_UNKNOWN) {
            dl_go(dl, DL_SETTLING, now);
        } else if (phase == DL_PHASE_LANDING) {
            dl_go(dl, DL_ABORT, now);
        } else if (geom_ok && dl->datum_set) {
            float dev = in->clearance_m - dl->datum_clearance_m;
            if (fabsf(dev) > DL_TERRAIN_THRESH_M) {
                dl->rejected_count++;
                dl_go(dl, DL_TERRAIN_SUSPECT, now);
                reject = true;              /* FIRST-SAMPLE rejection */
            }
        }
        break;

    case DL_VERTICAL_TRANSITION:
        if (phase == DL_PHASE_HOLD_LAYER) {
            dl_go(dl, DL_SETTLING, now);
        } else if (dl_elapsed(now, dl->t_state_ms) > DL_TRANSITION_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        break;

    case DL_SETTLING: {
        bool quiet = dl_finite(in->vz_cmd_mps) && dl_finite(in->vz_est_mps) &&
                     dl_finite(in->vxy_mps) && dl_finite(in->tilt_deg) &&
                     fabsf(in->vz_cmd_mps) < DL_VZ_CMD_EPS &&
                     fabsf(in->vz_est_mps) < DL_VZ_EST_EPS &&
                     fabsf(in->vxy_mps) < DL_XY_EPS &&
                     in->tilt_deg <= DL_TILT_MAX_DEG;
        if (quiet) {
            if (!dl->t_stable_set) {
                dl->t_stable_set = true;
                dl->t_stable_ms = now;
            } else if (dl_elapsed(now, dl->t_stable_ms) >= DL_SETTLE_DWELL_MS) {
                dl_go(dl, DL_DATUM_REVALIDATE, now);
                break;
            }
        } else {
            dl->t_stable_set = false;
        }
        if (dl_elapsed(now, dl->t_state_ms) > DL_TRANSITION_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        break;
    }

    case DL_DATUM_REVALIDATE: {
        bool decided = false;
        if (!geom_ok) {
            dl_window_clear(dl);
            dl->t_stable_set = false;
        } else if (dl_window_push_stable(dl, in->clearance_m)) {
            if (!dl->t_stable_set) {
                dl->t_stable_set = true;
                dl->t_stable_ms = now;
                decided = true;                 /* oracle returns here */
            } else if (dl_elapsed(now, dl->t_stable_ms) <
                       DL_REVALIDATE_DWELL_MS) {
                decided = true;                 /* oracle returns here */
            } else {
                float median = dl_window_median(dl);
                if (!dl->expected_set || !dl_finite(median)) {
                    dl_go(dl, DL_ABORT, now);
                } else if (fabsf(median - dl->expected_clearance_m) <=
                           DL_REVALIDATE_TOL_M) {
                    dl->datum_clearance_m = median;
                    dl->datum_set = true;
                    dl->expected_set = false;
                    dl->expected_clearance_m = NAN;
                    dl->datum_epoch++;
                    dl_go(dl, DL_DATUM_VALID, now);
                } else {
                    /* stable but WRONG: an obstacle, not the datum floor */
                    dl_go(dl, DL_ABORT, now);
                }
                decided = true;
            }
        }
        if (!decided &&
            dl_elapsed(now, dl->t_state_ms) > DL_REVALIDATE_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        break;
    }

    case DL_TERRAIN_SUSPECT:
        dl_go(dl, DL_BRAKE, now);
        reject = true;
        break;

    case DL_BRAKE:
        if (dl_finite(in->vxy_mps) && fabsf(in->vxy_mps) < DL_XY_EPS) {
            dl_go(dl, DL_RETREAT, now);
        } else if (dl_elapsed(now, dl->t_state_ms) > DL_ABORT_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        reject = true;
        break;

    case DL_RETREAT:
        if (geom_ok && dl->datum_set &&
            fabsf(in->clearance_m - dl->datum_clearance_m) <=
                DL_TERRAIN_THRESH_M) {
            if (dl_window_push_stable(dl, in->clearance_m)) {
                dl_go(dl, DL_RECOVER, now);
            }
        } else {
            dl_window_clear(dl);
        }
        if (dl->state == DL_RETREAT &&
            dl_elapsed(now, dl->t_state_ms) > DL_ABORT_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        reject = true;
        break;

    case DL_RECOVER:
        if (geom_ok && dl_window_push_stable(dl, in->clearance_m)) {
            dl_go(dl, DL_DATUM_VALID, now);
        } else if (dl_elapsed(now, dl->t_state_ms) > DL_ABORT_TIMEOUT_MS) {
            dl_go(dl, DL_ABORT, now);
        }
        break;

    default:
        break;
    }

    dl_publish(dl, reject, out);
}
