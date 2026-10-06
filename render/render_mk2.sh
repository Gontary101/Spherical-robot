#!/usr/bin/env bash
# Mk2 figure set. Usage: BLENDER=/path/to/blender render/render_mk2.sh
set -e
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
R="$B -b -P render/render_gyra.py -- --cad out2"
$R --scene hero          --out media/mk2_hero.png          --res 1920x1200 --samples 96
$R --scene cutaway       --out media/mk2_cutaway.png       --res 1920x1200 --samples 96
$R --scene cutaway       --out media/mk2_cutaway_posed.png --res 1600x1000 --samples 64 --pitch 60 --bob 35
$R --scene exploded      --out media/mk2_exploded.png      --res 1920x1200 --samples 64
$R --scene section_front --out media/mk2_section_front.png --res 1600x1400 --samples 64 --ortho 0.78
$R --scene pose          --out media/mk2_leaning_turn.png  --res 1600x1000 --samples 64 --roll 10 --yaw -28 --pitch 20 --bob 35
python3 render/annotate.py media/mk2_cutaway.png media/mk2_cutaway_annotated.png title="GYRA Mk2" ring drive pinion bob battery axle bay level rollers tyre yoke
