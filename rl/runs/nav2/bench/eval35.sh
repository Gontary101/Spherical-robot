cd /home/user/Spherical-robot
export GYRA_BASE=model GYRA_KSLOPE=2.46
B=rl/runs/nav2/bench
for c in m005M m010M final; do
  echo "=== $c"
  GYRA_LOW=$PWD/rl/runs/loco35/$c.pt python rl/stop_ctrl.py --sweep-cascade --policy-only 2>&1 | grep -v Warn
  GYRA_LOW=$PWD/rl/runs/loco35/$c.pt python rl/stop_ctrl.py --nav-bench '[]' 2>&1 | grep -v Warn
done
sed -i 's#rl/runs/loco31/best.pt"#rl/runs/loco35/final.pt"#' /tmp/claude-0/-home-user-Spherical-robot/8285b438-4880-50ed-a2d1-c190b5b65d14/scratchpad/slope_hold.py 2>/dev/null
GYRA_LOW=$PWD/rl/runs/loco35/final.pt python /tmp/claude-0/-home-user-Spherical-robot/8285b438-4880-50ed-a2d1-c190b5b65d14/scratchpad/slope_hold.py 2>&1 | grep -v Warn
python rl/lowlevel_pipeline.py --name loco35 --low rl/runs/loco35/final.pt --base model
python rl/lowlevel_pipeline.py --compare loco31_cd2 loco35
for s in 100 0; do
  GYRA_SEED0=$s GYRA_SENSORS=asbuilt_lane GYRA_REACH=1 GYRA_SYSID=$B/sysid_loco35_pipe.json GYRA_RESP=$B/resp_loco35.json \
    python rl/diagnose_nav.py --nav mpc --shield --episodes 16 --procs 4 --low $PWD/rl/runs/loco35/final.pt --scenarios "slope 8 deg" "slope 12 deg" --out $B/slopes_loco35_s$s.json > $B/slopes_loco35_s$s.log 2>&1
done
python3 - <<'PY'
import json
E=json.load(open('rl/runs/nav2/bench/slopes_loco35_s100.json'))+json.load(open('rl/runs/nav2/bench/slopes_loco35_s0.json'))
for sc in ('slope 8 deg','slope 12 deg'):
    X=[e for e in E if e['scenario']==sc]; C=[e for e in X if e.get('failure')=='collision']
    print('loco35', sc, 'success', sum(e['success'] for e in X), '/', len(X), 'collisions', len(C), 'at fault', sum(bool(e.get('at_fault')) for e in C), 'timeouts', sum(e.get('failure')=='timeout' for e in X), 'falls', sum(e.get('failure')=='fall' for e in X))
PY
echo ALLDONE
