#!/usr/bin/env bash
# Copies the generated release package (cad/build/out/release, gitignored) into cad/build/release/ for the repo:
# drawings (PDF), plate profiles (DXF), printed parts (STL), BOM / calcs / step table / check reports, and the
# STEP files zipped. Run after: assemble.py, checks.py, assembly.py, calcs.py, bom.py, guide.py, export.py.
set -euo pipefail
cd "$(dirname "$0")"
OUT=out
REL=release
rm -rf "$REL"
mkdir -p "$REL"
cp -r "$OUT/release/drawings" "$OUT/release/dxf" "$OUT/release/stl" "$REL/"
for f in bom.md bom.csv calcs.md cots.md assembly_steps.md checks.json assembly_check.json mass_properties.json \
         sim_build_masses.json yoke_range_build.json actuator_loads.json; do
  [ -f "$OUT/$f" ] && cp "$OUT/$f" "$REL/"
done
(cd "$OUT/release" && zip -q -9 "../../$REL/gyra_mk21_build_assembly_step.zip" gyra_mk21_build.step)
(cd "$OUT/release" && zip -q -9 -r "../../$REL/gyra_mk21_build_parts_step.zip" parts)
du -sh "$REL"/* | sort -h
