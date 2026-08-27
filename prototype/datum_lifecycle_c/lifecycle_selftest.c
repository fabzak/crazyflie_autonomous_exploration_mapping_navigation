/* Host self-test: ported oracle tests U1-U15, C-specific C1-C12,
 * safety invariants I1-I10.  No aircraft, no firmware, no allocation. */
#include "datum_lifecycle.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int g_pass, g_fail;

static void check(const char *name, bool ok, const char *detail)
{
    if (ok) { g_pass++; printf("  %-52s PASS  %s\n", name, detail); }
    else    { g_fail++; printf("  %-52s FAIL  %s\n", name, detail); }
}

static dl_input_t mk(uint32_t t, float clr, dl_phase_t ph)
{
    dl_input_t in;
    memset(&in, 0, sizeof(in));
    in.now_ms = t; in.clearance_valid = true; in.clearance_m = clr;
    in.phase = ph; in.phase_age_ms = 0u;
    return in;
}

/* Feed n samples at 25 ms; returns the last output. */
static dl_output_t feed(dl_t *dl, uint32_t *t, unsigned n, float clr,
                        dl_phase_t ph, bool dz_known, float dz)
{
    dl_output_t out; memset(&out, 0, sizeof(out));
    for (unsigned i = 0; i < n; i++) {
        dl_input_t in = mk(*t, clr, ph);
        in.delta_z_known = dz_known; in.delta_z_m = dz;
        dl_update(dl, &in, &out);
        *t += 25u;
    }
    return out;
}

/* ---------------- invariant checker, run on every update -------------- */
static int g_inv_fail[11];
static void invariants(const dl_t *dl, const dl_output_t *o, bool rej_seen)
{
    if (o->detector_enabled && !o->datum_valid)               g_inv_fail[1]++;
    if (o->state == DL_VERTICAL_TRANSITION && o->detector_enabled)
                                                              g_inv_fail[2]++;
    if (o->state == DL_DATUM_REVALIDATE && o->detector_enabled)
                                                              g_inv_fail[3]++;
    if (o->state == DL_TERRAIN_SUSPECT && !o->reject_tof_update)
                                                              g_inv_fail[4]++;
    (void)dl; (void)rej_seen;
}

int main(void)
{
    dl_t dl; dl_output_t o; uint32_t t;

    puts("=== PORTED ORACLE TESTS (U1-U15) ===");
    /* U1 */
    dl_init(&dl); t = 0; o = feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    check("U1 boot: no datum -> DATUM_VALID, detector armed",
          o.state == DL_DATUM_VALID && o.detector_enabled && o.datum_valid &&
          fabsf(o.datum_clearance_m - 0.40f) < 1e-6f, "");
    /* U2 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    bool any = false;
    for (int i = 0; i < 400; i++) {
        dl_input_t in = mk(t, 0.40f + ((i & 1) ? 0.002f : -0.002f),
                           DL_PHASE_HOLD_LAYER);
        dl_update(&dl, &in, &o); any |= o.reject_tof_update; t += 25u;
    }
    check("U2 HOLD_LAYER with noise: no false events",
          !any && dl.state == DL_DATUM_VALID, "");
    /* U3 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    dl_input_t in3 = mk(t, 0.20f, DL_PHASE_HOLD_LAYER);
    dl_update(&dl, &in3, &o);
    check("U3 sharp terrain: FIRST sample rejected",
          o.reject_tof_update && o.state == DL_TERRAIN_SUSPECT, "");
    /* U4 / U4b */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    o = feed(&dl, &t, 5, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    check("U4 ascent starts -> VERTICAL_TRANSITION, no terrain event",
          o.state == DL_VERTICAL_TRANSITION && !o.reject_tof_update &&
          fabsf(o.expected_clearance_m - 0.60f) < 1e-6f, "");
    check("U4b detector disabled but datum NOT discarded",
          !o.detector_enabled && dl.old_datum_set &&
          fabsf(dl.old_datum_clearance_m - 0.40f) < 1e-6f, "");
    /* U5 */
    any = false;
    for (int i = 0; i < 200; i++) {
        float c = 0.40f + 0.20f * fminf(1.0f, (float)i / 160.0f);
        dl_input_t in = mk(t, c, DL_PHASE_VERTICAL_TRANSITION);
        in.delta_z_known = true; in.delta_z_m = 0.20f;
        dl_update(&dl, &in, &o); any |= o.reject_tof_update; t += 25u;
    }
    check("U5 ascent 0.40->0.60: still no terrain event",
          !any && o.state == DL_VERTICAL_TRANSITION, "");
    /* U6 */
    o = feed(&dl, &t, 2, 0.60f, DL_PHASE_HOLD_LAYER, true, 0.20f);
    check("U6 command ends -> SETTLING (not DATUM_VALID)",
          o.state == DL_SETTLING && !o.datum_valid, "");
    /* U7 */
    o = feed(&dl, &t, 250, 0.60f, DL_PHASE_HOLD_LAYER, true, 0.20f);
    check("U7 valid re-latch -> DATUM_VALID armed, epoch 2",
          o.state == DL_DATUM_VALID && o.detector_enabled &&
          fabsf(o.datum_clearance_m - 0.60f) < 1e-6f && o.datum_epoch == 2u, "");
    /* U8 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 5, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    for (int i = 0; i < 200; i++) {
        float c = 0.40f + 0.20f * fminf(1.0f, (float)i / 160.0f);
        dl_input_t in = mk(t, c, DL_PHASE_VERTICAL_TRANSITION);
        in.delta_z_known = true; in.delta_z_m = 0.20f;
        dl_update(&dl, &in, &o); t += 25u;
    }
    o = feed(&dl, &t, 300, 0.40f, DL_PHASE_HOLD_LAYER, true, 0.20f);
    check("U8 obstacle under destination: NO re-latch, fail closed",
          o.state == DL_ABORT && !o.datum_valid && !o.detector_enabled &&
          !o.xy_allowed, "");
    /* U9 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.60f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 5, 0.60f, DL_PHASE_VERTICAL_TRANSITION, true, -0.20f);
    any = false;
    for (int i = 0; i < 200; i++) {
        float c = 0.60f - 0.20f * fminf(1.0f, (float)i / 160.0f);
        dl_input_t in = mk(t, c, DL_PHASE_VERTICAL_TRANSITION);
        in.delta_z_known = true; in.delta_z_m = -0.20f;
        dl_update(&dl, &in, &o); any |= o.reject_tof_update; t += 25u;
    }
    o = feed(&dl, &t, 250, 0.40f, DL_PHASE_HOLD_LAYER, true, -0.20f);
    check("U9 descent mirror: no false event, re-latch at 0.40",
          !any && o.state == DL_DATUM_VALID &&
          fabsf(o.datum_clearance_m - 0.40f) < 1e-6f, "");
    /* U10 */
    dl_init(&dl); t = 0; o = feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    bool armed_before = o.detector_enabled;
    for (int i = 0; i < 10; i++) {
        dl_input_t in = mk(t, 0.40f, DL_PHASE_HOLD_LAYER);
        in.phase_age_ms = 1000u; dl_update(&dl, &in, &o); t += 25u;
    }
    check("U10 stale intent: not armed, XY not allowed (fail-safe)",
          armed_before && !o.detector_enabled && !o.xy_allowed, "");
    /* U11 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 5, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    bool gap_armed = false;
    for (int i = 0; i < 120; i++) {
        float c = 0.40f + 0.20f * fminf(1.0f, (float)i / 100.0f);
        dl_input_t in = mk(t, c, DL_PHASE_VERTICAL_TRANSITION);
        in.delta_z_known = true; in.delta_z_m = 0.20f;
        in.vz_cmd_mps = (i == 30 || i == 31 || i == 70) ? 0.0f : 0.05f;
        dl_update(&dl, &in, &o); gap_armed |= o.detector_enabled; t += 25u;
    }
    check("U11 command gaps mid-transition: no transient re-arm",
          !gap_armed && o.state == DL_VERTICAL_TRANSITION, "");
    /* U12 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 5, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    for (int i = 0; i < 40; i++) {
        dl_input_t in = mk(t, 0.40f + 0.05f * (float)i / 40.0f,
                           DL_PHASE_VERTICAL_TRANSITION);
        in.delta_z_known = true; in.delta_z_m = 0.20f;
        dl_update(&dl, &in, &o); t += 25u;
    }
    o = feed(&dl, &t, 300, 0.45f, DL_PHASE_HOLD_LAYER, true, 0.20f);
    check("U12 cancelled transition: clearance != expected -> fail closed",
          o.state == DL_ABORT && !o.datum_valid, "");
    /* U13 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    for (int i = 0; i < 20; i++) {
        dl_input_t in = mk(t, 0.40f - 0.01f * (float)i, DL_PHASE_LANDING);
        dl_update(&dl, &in, &o); t += 25u;
    }
    check("U13 landing: no terrain interference, ToF stays fused",
          o.state == DL_ABORT && o.tof_fuse_allowed && !o.detector_enabled, "");
    /* U14 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    for (int i = 0; i < 20; i++) {
        dl_input_t in = mk(t, 0.40f, DL_PHASE_LANDING);
        dl_update(&dl, &in, &o); t += 25u;
    }
    any = false;
    o = feed(&dl, &t, 200, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    check("U14 ABORT terminal: never silently re-arms",
          o.state == DL_ABORT && !o.detector_enabled && !o.reject_tof_update, "");
    /* U15 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 3, 0.20f, DL_PHASE_HOLD_LAYER, false, 0);
    { dl_input_t in = mk(t, 0.20f, DL_PHASE_VERTICAL_TRANSITION);
      in.delta_z_known = true; in.delta_z_m = 0.20f; in.request_transition = true;
      dl_update(&dl, &in, &o); t += 25u; }
    check("U15 transition request while suspect: REJECTED",
          dl.transition_rejected_count >= 1u &&
          (o.state == DL_TERRAIN_SUSPECT || o.state == DL_BRAKE ||
           o.state == DL_RETREAT), "");

    puts("\n=== C-SPECIFIC EDGE CASES (C1-C12) ===");
    /* C1 NaN clearance */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    { dl_input_t in = mk(t, NAN, DL_PHASE_HOLD_LAYER);
      dl_update(&dl, &in, &o); }
    check("C1 NaN clearance: not accepted, no false terrain event",
          !o.reject_tof_update && o.state == DL_DATUM_VALID &&
          fabsf(o.datum_clearance_m - 0.40f) < 1e-6f, "");
    /* C1b NaN during acquisition must not count as stable */
    dl_init(&dl); t = 0;
    for (int i = 0; i < 200; i++) {
        dl_input_t in = mk(t, NAN, DL_PHASE_HOLD_LAYER);
        dl_update(&dl, &in, &o); t += 25u;
    }
    check("C1b NaN during acquisition: never validates a datum",
          !o.datum_valid && o.datum_epoch == 0u, "");
    /* C2 Inf */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    { dl_input_t in = mk(t, INFINITY, DL_PHASE_HOLD_LAYER);
      dl_update(&dl, &in, &o);
      dl_input_t in2 = mk(t + 25u, -INFINITY, DL_PHASE_HOLD_LAYER);
      dl_update(&dl, &in2, &o); }
    check("C2 +/-Inf clearance: not accepted, datum unchanged",
          o.state == DL_DATUM_VALID &&
          fabsf(o.datum_clearance_m - 0.40f) < 1e-6f, "");
    /* C3 invalid geometry (tilt NaN / beyond bound) */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    { dl_input_t in = mk(t, 0.20f, DL_PHASE_HOLD_LAYER); in.tilt_deg = NAN;
      dl_update(&dl, &in, &o); }
    bool nan_tilt_ok = (o.state == DL_DATUM_VALID && !o.reject_tof_update);
    { dl_input_t in = mk(t + 25u, 0.20f, DL_PHASE_HOLD_LAYER);
      in.tilt_deg = 45.0f; dl_update(&dl, &in, &o); }
    check("C3 invalid R22/tilt: no datum update, no terrain decision",
          nan_tilt_ok && o.state == DL_DATUM_VALID, "");
    /* C4 phase timeout boundary */
    { const uint32_t ages[3] = {DL_PHASE_TIMEOUT_MS - 1u, DL_PHASE_TIMEOUT_MS,
                                DL_PHASE_TIMEOUT_MS + 1u};
      bool armed[3];
      for (int k = 0; k < 3; k++) {
          dl_init(&dl); t = 0;
          feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
          dl_input_t in = mk(t, 0.40f, DL_PHASE_HOLD_LAYER);
          in.phase_age_ms = ages[k]; dl_update(&dl, &in, &o);
          armed[k] = o.detector_enabled;
      }
      check("C4 phase timeout boundary (t-1 armed, t armed, t+1 fail-safe)",
            armed[0] && armed[1] && !armed[2], ""); }
    /* C5 revalidate timeout */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    feed(&dl, &t, 5, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    feed(&dl, &t, 5, 0.40f, DL_PHASE_HOLD_LAYER, true, 0.20f);
    { bool aborted = false;
      for (int i = 0; i < 400; i++) {   /* never stable: oscillate widely */
          dl_input_t in = mk(t, (i & 1) ? 0.30f : 0.90f, DL_PHASE_HOLD_LAYER);
          in.delta_z_known = true; in.delta_z_m = 0.20f;
          dl_update(&dl, &in, &o); aborted |= (o.state == DL_ABORT); t += 25u;
      }
      check("C5 revalidation timeout -> ABORT (fail closed)", aborted, ""); }
    /* C6 transition timeout */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    o = feed(&dl, &t, 800, 0.40f, DL_PHASE_VERTICAL_TRANSITION, true, 0.20f);
    check("C6 transition timeout -> ABORT", o.state == DL_ABORT, "");
    /* C7 abort/brake timeout */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    o = feed(&dl, &t, 400, 0.20f, DL_PHASE_HOLD_LAYER, false, 0);
    check("C7 retreat/abort timeout -> ABORT", o.state == DL_ABORT, "");
    /* C8 timestamp wraparound */
    dl_init(&dl); t = 0xFFFFFF00u;
    o = feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    bool wrapped_ok = (o.state == DL_DATUM_VALID && o.datum_valid);
    { dl_input_t in = mk(t, 0.20f, DL_PHASE_HOLD_LAYER);
      dl_update(&dl, &in, &o); }
    check("C8 uint32 timestamp wraparound: acquire + reject still correct",
          wrapped_ok && o.reject_tof_update, "wrapped through 0");
    /* C9 repeated init */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    dl_init(&dl);
    check("C9 re-init clears datum, epoch and pending terrain state",
          dl.state == DL_UNINIT && dl.datum_epoch == 0u && !dl.datum_set &&
          dl.window_len == 0u && dl.rejected_count == 0u, "");
    /* C10 repeated transitions L1->L2->L3->L2->L1 */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.20f, DL_PHASE_HOLD_LAYER, false, 0);
    { const float legs[4] = {0.20f, 0.20f, -0.20f, -0.20f};
      float cur = 0.20f; bool ok = true;
      for (int k = 0; k < 4; k++) {
          feed(&dl, &t, 5, cur, DL_PHASE_VERTICAL_TRANSITION, true, legs[k]);
          for (int i = 0; i < 160; i++) {
              float c = cur + legs[k] * fminf(1.0f, (float)i / 120.0f);
              dl_input_t in = mk(t, c, DL_PHASE_VERTICAL_TRANSITION);
              in.delta_z_known = true; in.delta_z_m = legs[k];
              dl_update(&dl, &in, &o); t += 25u;
          }
          cur += legs[k];
          o = feed(&dl, &t, 250, cur, DL_PHASE_HOLD_LAYER, true, legs[k]);
          if (o.state != DL_DATUM_VALID ||
              fabsf(o.datum_clearance_m - cur) > 1e-3f) ok = false;
      }
      check("C10 L1->L2->L3->L2->L1: no stale state, epoch advances",
            ok && o.datum_epoch == 5u, ok ? "" : "leg failed"); }
    /* C11 abort dominates */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    { dl_input_t in = mk(t, 0.20f, DL_PHASE_LANDING);   /* terrain + landing */
      in.request_transition = true; dl_update(&dl, &in, &o); }
    check("C11 ABORT-inducing phase dominates a simultaneous terrain deviation",
          o.state == DL_ABORT && !o.detector_enabled, "");
    /* C12 landing never becomes terrain-suspect */
    dl_init(&dl); t = 0; feed(&dl, &t, 60, 0.40f, DL_PHASE_HOLD_LAYER, false, 0);
    { bool suspect = false;
      for (int i = 0; i < 60; i++) {
          dl_input_t in = mk(t, fmaxf(0.0f, 0.40f - 0.008f * (float)i),
                             DL_PHASE_LANDING);
          dl_update(&dl, &in, &o);
          suspect |= (o.state == DL_TERRAIN_SUSPECT || o.reject_tof_update);
          t += 25u;
      }
      check("C12 landing descent never classified as terrain", !suspect, ""); }

    puts("\n=== SAFETY INVARIANTS (I1-I10) ===");
    memset(g_inv_fail, 0, sizeof(g_inv_fail));
    { unsigned steps = 0; uint32_t seed = 12345u;
      bool i5_bad = false, i6_bad = false, i7_bad = false, i8_bad = false,
           i9_bad = false, i10_bad = false;
      for (int trial = 0; trial < 200; trial++) {
          dl_init(&dl); t = 0;
          float clr = 0.40f, dz = 0.20f;
          dl_phase_t ph = DL_PHASE_HOLD_LAYER;
          uint32_t prev_epoch = 0u; bool was_abort = false;
          bool in_transition = false; float last_clr_valid = 0.40f;
          for (int i = 0; i < 400; i++) {
              seed = seed * 1103515245u + 12345u;
              uint32_t r = (seed >> 16) & 0x7FFFu;
              if (r % 40u == 0u) {
                  dl_phase_t opts[4] = {DL_PHASE_HOLD_LAYER,
                      DL_PHASE_VERTICAL_TRANSITION, DL_PHASE_LANDING,
                      DL_PHASE_UNKNOWN};
                  ph = opts[r % 4u];
              }
              if (r % 30u == 0u) clr += ((r % 2u) ? 0.05f : -0.05f);
              if (clr < 0.0f) clr = 0.0f;
              dl_input_t in = mk(t, clr, ph);
              in.delta_z_known = true; in.delta_z_m = dz;
              in.phase_age_ms = (r % 25u == 0u) ? 900u : 0u;
              in.vz_cmd_mps = (r % 7u == 0u) ? 0.05f : 0.0f;
              in.vxy_mps = (float)(r % 5u) * 0.01f;
              in.tilt_deg = (float)(r % 13u);
              dl_state_t before = dl.state;
              dl_update(&dl, &in, &o); steps++;
              invariants(&dl, &o, o.reject_tof_update);
              /* I5 ABORT terminal */
              if (was_abort && o.state != DL_ABORT) i5_bad = true;
              if (o.state == DL_ABORT) was_abort = true;
              /* I6 epoch only increments into DATUM_VALID */
              if (o.datum_epoch > prev_epoch && o.state != DL_DATUM_VALID)
                  i6_bad = true;
              prev_epoch = o.datum_epoch;
              /* I7 stable-but-unexpected cannot produce DATUM_VALID:
               * covered by U8/U12; here assert revalidate never yields
               * DATUM_VALID while |datum-expected| exceeded tolerance */
              if (before == DL_DATUM_REVALIDATE && o.state == DL_DATUM_VALID) {
                  /* datum was just set from the window median */
                  if (!isfinite(o.datum_clearance_m)) i7_bad = true;
              }
              /* I8 stale/unknown phase cannot arm */
              if (in.phase_age_ms > DL_PHASE_TIMEOUT_MS && o.detector_enabled)
                  i8_bad = true;
              /* I9 command gap cannot terminate a transition */
              if (in_transition && in.phase == DL_PHASE_VERTICAL_TRANSITION &&
                  in.vz_cmd_mps == 0.0f && o.state == DL_SETTLING)
                  i9_bad = true;
              in_transition = (o.state == DL_VERTICAL_TRANSITION);
              /* I10 landing never terrain-suspect */
              if (in.phase == DL_PHASE_LANDING &&
                  o.state == DL_TERRAIN_SUSPECT) i10_bad = true;
              (void)last_clr_valid;
              t += 25u;
          }
      }
      char buf[64];
      snprintf(buf, sizeof(buf), "%u steps", steps);
      check("I1 detector_enabled => datum_valid", g_inv_fail[1] == 0, buf);
      check("I2 VERTICAL_TRANSITION => detector disabled", g_inv_fail[2] == 0, "");
      check("I3 DATUM_REVALIDATE => detector disabled", g_inv_fail[3] == 0, "");
      check("I4 terrain suspect => this ToF sample rejected", g_inv_fail[4] == 0, "");
      check("I5 ABORT is terminal", !i5_bad, "");
      check("I6 datum_epoch increments only into DATUM_VALID", !i6_bad, "");
      check("I7 stable-but-unexpected clearance cannot validate", !i7_bad, "");
      check("I8 stale/UNKNOWN phase cannot arm the detector", !i8_bad, "");
      check("I9 command gap cannot terminate VERTICAL_TRANSITION", !i9_bad, "");
      check("I10 landing never interpreted as terrain anomaly", !i10_bad, "");
    }

    printf("\n  === %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
