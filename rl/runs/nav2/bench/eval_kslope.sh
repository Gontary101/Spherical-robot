cd /home/user/Spherical-robot
B=rl/runs/nav2/bench
export GYRA_KSLOPE=2.46
for s in 100 0; do
  GYRA_SEED0=$s GYRA_SENSORS=asbuilt_lane GYRA_REACH=1 GYRA_SYSID=$B/sysid_loco31_pipe.json GYRA_RESP=$B/resp_loco31.json \
    python rl/diagnose_nav.py --nav mpc --shield --episodes 16 --procs 4 --scenarios "slope 8 deg" "slope 12 deg" --out $B/slopes_kslope2_s$s.json > $B/slopes_kslope2_s$s.log 2>&1
done
python3 - <<'PY'
import json
E=json.load(open('rl/runs/nav2/bench/slopes_kslope2_s100.json'))+json.load(open('rl/runs/nav2/bench/slopes_kslope2_s0.json'))
for sc in ('slope 8 deg','slope 12 deg'):
    X=[e for e in E if e['scenario']==sc]; C=[e for e in X if e.get('failure')=='collision']
    print('loco31+kslope2', sc, 'success', sum(e['success'] for e in X), '/', len(X), 'collisions', len(C), 'at fault', sum(bool(e.get('at_fault')) for e in C), 'timeouts', sum(e.get('failure')=='timeout' for e in X), 'falls', sum(e.get('failure')=='fall' for e in X))
PY
python rl/lowlevel_pipeline.py --name loco31 --low rl/runs/loco31/best.pt --base classical --tag kslope2
python rl/lowlevel_pipeline.py --compare loco31_cd2 loco31_kslope2
echo ALLDONE
