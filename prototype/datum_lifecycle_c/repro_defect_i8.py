import sys
sys.path.insert(0,'/home/fabijan/crazyflie_autonomous_exploration_mapping_navigation/prototype/datum_lifecycle_c')
import oracle_datum_lifecycle as O
lc=O.DatumLifecycle(); t=0.0
lc.update(t,0.40,O.HOLD_LAYER,0.0); t+=0.025      # fresh -> DATUM_ACQUIRE
print(f"  {'sample':>6} {'phase_age':>10} {'state':<18} {'detector':>9} {'xy_allowed':>11}")
for i in range(24):
    lc.update(t,0.40,O.HOLD_LAYER,1.0); t+=0.025   # ALL stale
    if i>=17:
        print(f"  {i:6d} {1.0:9.1f}s {lc.state:<18} "
              f"{str(lc.detector_enabled):>9} {str(lc.xy_allowed):>11}")
