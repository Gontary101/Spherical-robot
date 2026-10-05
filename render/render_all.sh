#!/usr/bin/env bash
# Renders every figure used in the docs. Usage: BLENDER=/path/to/blender render/render_all.sh
set -e
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
R="$B -b -P render/render_gyra.py --"
$R --scene hero          --out media/gyra_hero.png          --res 1920x1200 --samples 128
$R --scene side          --out media/gyra_side.png          --res 1600x1000 --samples 96
$R --scene pose          --out media/gyra_leaning_turn.png  --res 1600x1000 --samples 96 --roll 14 --yaw -28 --pitch 22 --bob 26
$R --scene cutaway       --out media/gyra_cutaway.png       --res 1920x1200 --samples 128
$R --scene cutaway       --out media/gyra_cutaway_posed.png --res 1600x1000 --samples 96 --pitch 35 --bob 24 --gimbal 45
$R --scene xray          --out media/gyra_xray.png          --res 1600x1000 --samples 96
$R --scene exploded      --out media/gyra_exploded.png      --res 1920x1200 --samples 96
$R --scene section_front --out media/gyra_section_front.png --res 1400x1400 --samples 96
$R --scene section_side  --out media/gyra_section_side.png  --res 1400x1400 --samples 96
$R --scene section_side  --out media/gyra_section_side_driving.png --res 1400x1400 --samples 64 --pitch 40
$R --scene top           --out media/gyra_top.png           --res 1400x1000 --samples 64
python3 render/annotate.py media/gyra_cutaway.png media/gyra_cutaway_annotated.png cmg gimbal ring drive pinion bob battery axle bay level rollers tyre yoke
python3 render/annotate.py media/gyra_cutaway_posed.png media/gyra_cutaway_posed_annotated.png cmg gimbal drive bob battery axle level yoke
