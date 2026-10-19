# Rendering the GYRA Mk2.1 film at high quality

The film in the repository (`media/gyra_mk21_film.mp4`) was made with preview-grade settings so that it
could render on a 4-core CPU in about a day:

* Blender ran at 1280x720, with 8–10 samples per pixel.
* Only every other frame was rendered; motion interpolation filled in the rest.
* The result was upscaled to 1080p and squeezed to 1.7 Mbit/s to fit under GitHub's 100 MB limit.

This guide renders the same film from the same data with final-quality settings: every frame,
full resolution, real light bounces, and a 1080p or 4K master.

Nothing has to be re-simulated. The field mission (`media/showcase.npz`), the safety-filter snapshot
(`media/filter_snapshot.npz`), the CAD meshes (`cad/out21/`) and the fonts (`render/fonts/`) are all
in the repository.

## 1. What you need

| tool | version | used for |
|---|---|---|
| Blender | 4.2 LTS (4.2.3 tested) | the CAD shots and the field mission (Cycles) |
| Python | 3.10 or newer, with `numpy`, `matplotlib`, `pillow` | the explainer scenes and the compositor |
| ffmpeg | 6.x, built with `libx264` (and `libx265` for HEVC) | decoding frame sequences and encoding the film |
| GPU (strongly recommended) | NVIDIA (OptiX or CUDA), AMD (HIP), Apple Silicon (Metal), Intel Arc (oneAPI) | Cycles rendering |

```bash
git clone https://github.com/Gontary101/Spherical-robot.git
cd Spherical-robot
pip install numpy matplotlib pillow
blender --version          # must be on PATH: Blender 4.2.x
ffmpeg -version | head -1
```

On macOS, Blender's command-line binary is `/Applications/Blender.app/Contents/MacOS/Blender`. Either
alias it as `blender` or put that path in front of each command below.

## 2. Pick a quality preset

| preset | Blender resolution | frames | samples (CAD / mission) | explainer | film |
|---|---|---|---|---|---|
| repo film (preview) | 1280x720 | every 2nd | 10 / 8 | 1080p | 1080p H.264, 1.7 Mbit/s |
| **very high** | 1920x1080 | every frame | 128 / 96, `HQ=1` | 1080p | 1080p H.264, CRF 14 |
| **ultra** | 3840x2160 | every frame | 256 / 128, `HQ=1` | 4K (`--scale 2`) | 4K HEVC 10-bit, CRF 16, plus a ProRes master |

The film has about 7,400 Blender frames when every frame is rendered:

* the four CAD shots: 1,656 frames (title 168, assembly 768, x-ray 576, roll 144);
* the mission: about 5,730 frames.

There are also 3,168 explainer frames, made with matplotlib on the CPU.

`HQ=1` raises the light bounces from 4/2/2/2 to 12/4/4/8 (total / diffuse / glossy / transmission).
It also tightens the adaptive-sampling noise threshold to 0.005. That matters most for the
see-through shell in the x-ray shot and for the soft shadows in the mission. Denoising
(OpenImageDenoise) stays on in every preset.

## 3. Time one frame before committing

Render a single frame of each kind with your chosen settings and multiply by the frame counts above.
Use a scratch folder:

```bash
export GYRA_DEVICE=OPTIX     # or CUDA / HIP / METAL / ONEAPI; leave unset for CPU
time blender -b -P render/robot_shots.py -- --shot xray --out /tmp/gyra_test/robot \
     --res 1920x1080 --samples 128 --hq --start 200 --end 201
time blender -b -P render/animate_showcase.py -- --out /tmp/gyra_test/mission \
     --res 1920x1080 --samples 96 --hq --only 2400
```

Each run prints `GYRA_DEVICE=...: no such GPU found, rendering on the CPU` if the GPU backend is
not available. Fix that before the long render.

Open the two PNGs and check the noise. If the x-ray shell or the grass looks grainy, raise the
samples. If a frame takes too long, the mission is the place to save: lower its samples first,
because the CAD shots are seen up close.

## 4. Render

Every renderer can be resumed: it skips frames that already exist, so you can stop it (Ctrl-C) and
start it again with the same command. Use a **new frames folder** for every quality preset.
Otherwise frames from an earlier preset would be kept and mixed in.

### Very high (1080p)

```bash
F=$HOME/gyra_frames_1080          # needs about 25 GB free

# 1) Blender: the CAD shots, then the mission (several hours on a modern GPU)
export GYRA_DEVICE=OPTIX
RES=1920x1080 STEP=1 SAMPLES_CAD=128 SAMPLES_MISSION=96 HQ=1 render/render_video.sh $F

# 2) explainer scenes (CPU, about 4 minutes per scene)
for s in move steer level arch filter; do
  python render/explainer.py --scene $s --out $F/explainer
done

# 3) compose: overlays, titles, the HUD and crossfades -> one 1080p film
python render/compose_final.py --frames $F --explainer $F/explainer \
  --out gyra_mk21_film_1080.mp4 --res 1080 --crf 14 --preset slower \
  --results '["Field run: 238 m, 17 waypoints, no collisions", "208 benchmark missions: 88.9 % reach the goal", "1.4 % contacts, 0.5 % at fault (pods and 150 N shoves counted)"]'
```

### Ultra (4K)

```bash
F=$HOME/gyra_frames_4k            # needs about 90 GB free

export GYRA_DEVICE=OPTIX
RES=3840x2160 STEP=1 SAMPLES_CAD=256 SAMPLES_MISSION=128 HQ=1 render/render_video.sh $F

for s in move steer level arch filter; do
  python render/explainer.py --scene $s --out $F/explainer --scale 2     # 3840x2160, same layout
done

RESULTS='["Field run: 238 m, 17 waypoints, no collisions", "208 benchmark missions: 88.9 % reach the goal", "1.4 % contacts, 0.5 % at fault (pods and 150 N shoves counted)"]'

# delivery file: 4K HEVC, 10-bit
python render/compose_final.py --frames $F --explainer $F/explainer \
  --out gyra_mk21_film_4k.mp4 --res 2160 --codec h265 --crf 16 --preset slow --results "$RESULTS"

# optional archive/edit master: ProRes 422 HQ (very large: about 40 GB for 7 minutes)
python render/compose_final.py --frames $F --explainer $F/explainer \
  --out gyra_mk21_film_4k_master.mov --res 2160 --codec prores --results "$RESULTS"
```

The compositor draws every overlay in 1080p layout units and scales it to the output size. At 4K,
text, panels, callouts and the HUD are therefore drawn natively sharp, not upscaled. It also detects
the frame rate of each sequence on its own:

* frames rendered with `STEP=1` are used as they are;
* frames rendered with `STEP=2` get motion-compensated interpolation.

### Splitting the work

* **Render only some shots:** `SHOTS="assembly xray" RES=... render/render_video.sh $F`.
  Valid names are `title assembly xray roll mission`.
* **Several GPUs or machines:** give each machine a frame range of the mission and copy the PNGs
  into one folder. The frames are numbered by their 24 fps index:
  ```bash
  blender -b -P render/animate_showcase.py -- --out $F/mission --res 3840x2160 --samples 128 --hq --step 1 --start 0 --end 2900
  blender -b -P render/animate_showcase.py -- --out $F/mission --res 3840x2160 --samples 128 --hq --step 1 --start 2900
  ```
  The same `--start/--end` options exist for `render/robot_shots.py` (one `--shot` at a time).
* **Preview a single segment** before composing everything:
  `python render/compose_final.py ... --only assembly,xray --out preview.mp4`.
  The segment names are title, assembly, xray, roll, move, steer, level, arch, filter, card_field,
  mission and card_end.

## 5. Check the result

```bash
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate,codec_name -of compact gyra_mk21_film_4k.mp4
# expect: 3840x2160, 24/1, duration 417.5 s (6 min 58 s)
ffmpeg -ss 300 -i gyra_mk21_film_4k.mp4 -frames:v 1 check_300s.png    # a mission frame with the HUD
```

## 6. Settings reference

| setting | where | effect |
|---|---|---|
| `RES` | `render_video.sh` | Blender resolution. Use 16:9; the compositor rescales to its `--res`. |
| `STEP` | `render_video.sh` | 1 = every frame. 2 = every other frame (half the time, interpolated). |
| `SAMPLES_CAD`, `SAMPLES_MISSION` | `render_video.sh` | Cycles maximum samples per pixel. Adaptive sampling stops earlier on clean pixels. |
| `HQ=1` / `--hq` | `render_video.sh`, both Blender scripts | final-quality bounces, adaptive threshold 0.005 |
| `GYRA_DEVICE` | environment | GPU backend (`OPTIX`, `CUDA`, `HIP`, `METAL`, `ONEAPI`); unset = CPU |
| `--scale` | `render/explainer.py` | explainer resolution (1 = 1080p, 2 = 4K) |
| `--res` | `render/compose_final.py` | output height: 1080, 1440 or 2160 |
| `--codec`, `--crf`, `--preset` | `render/compose_final.py` | `h264` (yuv420p), `h265` (10-bit) or `prores` (422 HQ, 10-bit). CRF: lower = better; 14–18 is visually lossless. |

Fonts: the overlays use Poppins in upright weights only (Light, Regular, Medium, SemiBold, Bold),
licensed under the OFL in `render/fonts/`. Greek symbols in the explainer equations fall back to
DejaVu Sans, also upright.

## 7. The Mk2.1 build film (assembly, exploded views, how it works)

`media/gyra_mk21_build_film.mp4` (4 min 16 s) is rendered by `render/build_film.py` from the build CAD
(`cad/build`). The meshes are generated from the CAD once, so this film also needs CadQuery.

```bash
git clone https://github.com/Gontary101/Spherical-robot.git
cd Spherical-robot
git checkout claude/pensive-bohr-u0sl8n
pip install cadquery numpy pillow opencv-python

# 1) meshes + part data from the build CAD (a few minutes) -> cad/build/out/film/
python render/film_data.py

# 2) frames on the GPU (resumable: rerun the same command after an interruption)
export GYRA_DEVICE=OPTIX          # NVIDIA RTX; CUDA for older NVIDIA, HIP for AMD, METAL for Apple Silicon
blender -b -P render/build_film.py -- --out render/out_final --res 1920x1080 --samples 32

# 3) overlay data for the in-between frames, then labels, captions and encoding (needs ffmpeg on PATH)
blender -b -P render/build_film.py -- --out render/out_final --res 1920x1080 --json-only
python render/compose_build_film.py --frames render/out_final --out media/gyra_mk21_build_film.mp4
```

Windows (PowerShell): set the GPU with `$env:GYRA_DEVICE="OPTIX"` and call Blender by its full path,
`& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b -P render/build_film.py -- ...`.

Options for `build_film.py`:

| option | default | effect |
|---|---|---|
| `--samples N` | 8 | samples per pixel (denoised); 32 is clean, 64 for the x-ray shots to be spotless |
| `--every-frame` | off | render all 6144 frames instead of ~4800 (the rest are synthesised from optical flow) |
| `--motion-px P` | 24 | in-between frames are rendered, not synthesised, wherever the picture moves more than P pixels between the rendered neighbours (fast camera moves, parts flying in); lower = more frames, fewer flow artefacts |
| `--res WxH` | 1920x1080 | `3840x2160` for 4K (pass the same `--res` to the compositor) |
| `--worker k/n` | 0/1 | split the frames over n processes or machines (copy the PNGs into one folder) |
| `--only seg,...` | all | render some segments only, e.g. `--only explode,assembly` |
| `--frames a,b,...` | all | render single frames, for checks |
| `--preview` | off | 2 samples, every 4th frame: checks the whole film in a few hours on a CPU |

Without a GPU, `render/render_build_film.sh <dir> 2 2` runs two CPU workers then the composition; on a
4-core CPU at 8 samples that is about two days.
