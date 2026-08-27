/* Reads a whitespace trace on stdin, emits one output record per step.
 * Used by differential.py to compare against the Python oracle. */
#include "datum_lifecycle.h"
#include <stdio.h>
#include <math.h>

int main(void)
{
    dl_t dl;
    dl_init(&dl);
    unsigned long now, age;
    int cvalid, phase, dknown, req;
    double clr, vzc, vze, vxy, tilt, dz;
    while (scanf("%lu %d %lf %d %lu %lf %lf %lf %lf %d %lf %d",
                 &now, &cvalid, &clr, &phase, &age, &vzc, &vze, &vxy,
                 &tilt, &dknown, &dz, &req) == 12) {
        dl_input_t in = {
            .now_ms = (uint32_t)now,
            .clearance_valid = cvalid != 0,
            .clearance_m = (float)clr,
            .phase = (dl_phase_t)phase,
            .phase_age_ms = (uint32_t)age,
            .vz_cmd_mps = (float)vzc,
            .vz_est_mps = (float)vze,
            .vxy_mps = (float)vxy,
            .tilt_deg = (float)tilt,
            .delta_z_known = dknown != 0,
            .delta_z_m = (float)dz,
            .request_transition = req != 0,
        };
        dl_output_t out;
        dl_update(&dl, &in, &out);
        printf("%s %d %d %d %d %.6f %.6f %u\n",
               dl_state_name(out.state), out.datum_valid ? 1 : 0,
               out.detector_enabled ? 1 : 0, out.transition_active ? 1 : 0,
               out.reject_tof_update ? 1 : 0,
               isfinite(out.datum_clearance_m) ? (double)out.datum_clearance_m
                                               : (double)NAN,
               isfinite(out.expected_clearance_m)
                   ? (double)out.expected_clearance_m : (double)NAN,
               out.datum_epoch);
    }
    return 0;
}
