"""Differential Python-oracle vs C equivalence harness.

Feeds an identical timestamped trace into oracle_datum_lifecycle.py and the
compiled ./trace_runner, then compares every safety-relevant output at every
step.  Enums/booleans must match EXACTLY; floats use a predeclared tolerance.
"""
import math, random, subprocess, sys
import oracle_datum_lifecycle as O

FLOAT_TOL = 1e-4                     # predeclared numerical tolerance
PHASE_ID = {O.UNKNOWN: 0, O.HOLD_LAYER: 1, O.VERTICAL_TRANSITION: 2,
            O.LANDING: 3}


def run_oracle(trace):
    lc = O.DatumLifecycle()
    out = []
    for s in trace:
        rej = lc.update(s['t_ms']/1000.0,
                        s['clr'] if s['valid'] else None,
                        s['phase'], s['age_ms']/1000.0,
                        vz_cmd=s['vz_cmd'], vz_est=s['vz_est'],
                        vxy=s['vxy'], tilt_deg=s['tilt'],
                        delta_z=s['dz'] if s['dz_known'] else None,
                        request_transition=s['req'])
        out.append((lc.state, bool(lc.datum_valid), bool(lc.detector_enabled),
                    bool(lc.transition_active), bool(rej),
                    lc.datum_clearance if lc.datum_clearance is not None
                    else float('nan'),
                    lc.expected_new if lc.expected_new is not None
                    else float('nan'),
                    lc.datum_epoch))
    return out


def run_c(trace, binary='./trace_runner'):
    lines = ''.join(
        f"{s['t_ms']} {1 if s['valid'] else 0} {s['clr']:.9f} "
        f"{PHASE_ID[s['phase']]} {s['age_ms']} {s['vz_cmd']:.9f} "
        f"{s['vz_est']:.9f} {s['vxy']:.9f} {s['tilt']:.9f} "
        f"{1 if s['dz_known'] else 0} {s['dz']:.9f} {1 if s['req'] else 0}\n"
        for s in trace)
    r = subprocess.run([binary], input=lines, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'trace_runner failed: {r.stderr}')
    out = []
    for ln in r.stdout.strip().split('\n'):
        f = ln.split()
        out.append((f[0], f[1] == '1', f[2] == '1', f[3] == '1', f[4] == '1',
                    float(f[5]), float(f[6]), int(f[7])))
    return out


FIELDS = ('state', 'datum_valid', 'detector_enabled', 'transition_active',
          'reject_tof', 'datum_clearance', 'expected_clearance', 'datum_epoch')


def compare(name, trace):
    po, co = run_oracle(trace), run_c(trace)
    if len(po) != len(co):
        return [(name, -1, 'length', len(po), len(co))]
    bad = []
    for i, (p, c) in enumerate(zip(po, co)):
        for k, (pv, cv) in enumerate(zip(p, c)):
            if k in (5, 6):
                if math.isnan(pv) and math.isnan(cv):
                    continue
                if math.isnan(pv) != math.isnan(cv) or abs(pv - cv) > FLOAT_TOL:
                    bad.append((name, i, FIELDS[k], pv, cv))
            elif pv != cv:
                bad.append((name, i, FIELDS[k], pv, cv))
    return bad


# ---- deterministic trace builders --------------------------------------
def step(t_ms, clr=0.40, valid=True, phase=O.HOLD_LAYER, age_ms=0,
         vz_cmd=0.0, vz_est=0.0, vxy=0.0, tilt=0.0, dz=0.0, dz_known=False,
         req=False):
    return dict(t_ms=t_ms, clr=clr, valid=valid, phase=phase, age_ms=age_ms,
                vz_cmd=vz_cmd, vz_est=vz_est, vxy=vxy, tilt=tilt, dz=dz,
                dz_known=dz_known, req=req)


def hold(n, t0, clr=0.40, **kw):
    return [step(t0 + 25*i, clr=clr, **kw) for i in range(n)]


def deterministic_traces():
    T = {}
    T['boot acquisition'] = hold(80, 0)
    T['sharp box'] = hold(40, 0) + hold(40, 1000, clr=0.20)
    asc = hold(40, 0)
    asc += [step(1000+25*i, clr=0.40+0.20*min(1.0, i/100.0),
                 phase=O.VERTICAL_TRANSITION, dz=0.20, dz_known=True,
                 vz_cmd=0.05, vz_est=0.05) for i in range(120)]
    asc += hold(200, 4000, clr=0.60, dz=0.20, dz_known=True)
    T['valid ascent'] = asc
    des = hold(40, 0, clr=0.60)
    des += [step(1000+25*i, clr=0.60-0.20*min(1.0, i/100.0),
                 phase=O.VERTICAL_TRANSITION, dz=-0.20, dz_known=True,
                 vz_cmd=-0.05, vz_est=-0.05) for i in range(120)]
    des += hold(200, 4000, clr=0.40, dz=-0.20, dz_known=True)
    T['valid descent'] = des
    gap = hold(40, 0)
    gap += [step(1000+25*i, clr=0.40+0.20*min(1.0, i/100.0),
                 phase=O.VERTICAL_TRANSITION, dz=0.20, dz_known=True,
                 vz_cmd=(0.0 if i in (30, 31, 70) else 0.05),
                 vz_est=0.05) for i in range(120)]
    gap += hold(200, 4000, clr=0.60, dz=0.20, dz_known=True)
    T['command gap'] = gap
    rnd = random.Random(1)
    noi = hold(40, 0)
    noi += [step(1000+25*i, clr=0.40+0.20*min(1.0, i/100.0)
                 + rnd.gauss(0, 0.002), phase=O.VERTICAL_TRANSITION,
                 dz=0.20, dz_known=True, vz_cmd=0.05, vz_est=0.05)
            for i in range(120)]
    noi += [step(4000+25*i, clr=0.60+rnd.gauss(0, 0.002), dz=0.20,
                 dz_known=True) for i in range(200)]
    T['transition + noise'] = noi
    inv = hold(40, 0)
    inv += [step(1000+25*i, clr=0.40+0.20*min(1.0, i/100.0),
                 valid=not (40 <= i < 60), phase=O.VERTICAL_TRANSITION,
                 dz=0.20, dz_known=True, vz_cmd=0.05, vz_est=0.05)
            for i in range(120)]
    inv += hold(200, 4000, clr=0.60, dz=0.20, dz_known=True)
    T['transition + invalid ToF'] = inv
    wrong = hold(40, 0)
    wrong += [step(1000+25*i, clr=0.40, phase=O.VERTICAL_TRANSITION,
                   dz=0.20, dz_known=True, vz_cmd=0.05, vz_est=0.05)
              for i in range(120)]
    wrong += hold(250, 4000, clr=0.40, dz=0.20, dz_known=True)
    T['stable wrong clearance'] = wrong
    after = hold(40, 0) + hold(120, 1000, clr=0.40,
                               phase=O.VERTICAL_TRANSITION, dz=0.20,
                               dz_known=True, vz_cmd=0.05, vz_est=0.05)
    after = hold(40, 0)
    after += [step(1000+25*i, clr=0.40+0.20*min(1.0, i/100.0),
                   phase=O.VERTICAL_TRANSITION, dz=0.20, dz_known=True,
                   vz_cmd=0.05, vz_est=0.05) for i in range(120)]
    after += hold(120, 4000, clr=0.60, dz=0.20, dz_known=True)
    after += hold(60, 7000, clr=0.40, dz=0.20, dz_known=True)
    T['terrain right after re-arm'] = after
    susp = hold(40, 0) + hold(10, 1000, clr=0.20)
    susp += hold(40, 1250, clr=0.20, phase=O.VERTICAL_TRANSITION, dz=0.20,
                 dz_known=True, req=True)
    T['transition request while suspect'] = susp
    T['landing'] = hold(40, 0) + [step(1000+25*i, clr=max(0.0, 0.40-0.01*i),
                                      phase=O.LANDING) for i in range(40)]
    T['stale phase'] = hold(40, 0) + hold(40, 1000, age_ms=1000)
    T['abort via acquire timeout'] = [
        step(25*i, clr=0.40 + (0.05 if i % 2 else -0.05)) for i in range(500)]
    return T


def randomized_traces(seed, count, length):
    rnd = random.Random(seed)
    phases = [O.HOLD_LAYER, O.VERTICAL_TRANSITION, O.LANDING, O.UNKNOWN]
    out = {}
    for k in range(count):
        tr, t, clr, phase, dz = [], 0, 0.40, O.HOLD_LAYER, 0.20
        for _ in range(length):
            if rnd.random() < 0.03:
                phase = rnd.choice(phases)
            if rnd.random() < 0.05:
                clr += rnd.choice([-0.20, -0.05, -0.02, 0.02, 0.05, 0.20])
                clr = max(0.0, min(1.5, clr))
            if rnd.random() < 0.02:
                dz = rnd.choice([0.20, -0.20, 0.40, -0.40])
            tr.append(step(t, clr=clr + rnd.gauss(0, 0.003),
                           valid=rnd.random() > 0.05, phase=phase,
                           age_ms=rnd.choice([0, 0, 0, 200, 600, 1000]),
                           vz_cmd=rnd.choice([0.0, 0.0, 0.05, -0.05]),
                           vz_est=rnd.gauss(0, 0.03),
                           vxy=abs(rnd.gauss(0, 0.05)),
                           tilt=abs(rnd.gauss(0, 4.0)),
                           dz=dz, dz_known=rnd.random() > 0.1,
                           req=rnd.random() < 0.02))
            t += 25
        out[f'random[{seed}]#{k}'] = tr
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'det'
    if mode == 'det':
        traces = deterministic_traces()
    else:
        traces = randomized_traces(int(sys.argv[2]), int(sys.argv[3]),
                                   int(sys.argv[4]))
    total_steps = 0
    all_bad = []
    for name, tr in traces.items():
        total_steps += len(tr)
        bad = compare(name, tr)
        all_bad += bad
        if mode == 'det':
            print(f"  {name:<34} {len(tr):5d} steps  "
                  f"{'OK' if not bad else 'DIVERGE ' + str(len(bad))}")
    print(f"\n  traces={len(traces)}  steps={total_steps}  "
          f"divergences={len(all_bad)}")
    for b in all_bad[:8]:
        print(f"    {b}")
    sys.exit(1 if all_bad else 0)
