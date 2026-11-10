# GAIA dataset preprocessing (FFHQ)

This doc describes the process to turn raw FFHQ datasets into `FFHQ_512.zip`, which is consumed by `gaia/train.py`.

## 1. The data structure

```
FFHQ_512.zip
├── 00000/
│   ├── img00000000.png     512×512 RGBA
│   ├── img00000001.png
│   └── ...                 1000 images per folder
├── 00001/
├── ...                     140 folders
└── dataset.json
```

**Images** are 512×512 RGBA PNGs: RGB from the EG3D crop, alpha from MODNet. They are
renumbered `img%08d.png` in packing order and bucketed 1000 per folder — original FFHQ
ids are not preserved.

**`dataset.json`** is `{"labels": [[<path>, <380 floats>], ...]}`, one entry per image:

```
label = cam_pose(16) + intrinsics(9) + shape(300) + expression(50) + jaw(3) + eyelid(2)
        └──── EG3D camera, 25 ────┘   └────────── SMIRK FLAME, 355 ──────────┘
```

**Every face appears twice**, adjacent: the original and a horizontally-mirrored copy
(`img00000000` and `img00000001` are the same person). The mirror keeps the FLAME
parameters unchanged and carries a yaw-flipped camera.

**Count:** 69,957 usable FFHQ images × 2 = 139,914 entries. (FFHQ ships 70,000, but 43
are absent from EG3D's `cropping_params.json`.) GAIA's own released set has 139,848 — a
further **33 images** were excluded, listed in `missing_from_release.txt`. They are not
processing failures; this pipeline produces FLAME parameters for all 69,957. Drop those 33
ids (and their mirrors) if you want a set that matches the released one exactly.

## 2. Stage 1 — align + crop (EG3D upstream)

1. Get the FFHQ in-the-wild originals from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
2. Run [NVlabs/eg3d](https://github.com/NVlabs/eg3d)'s `dataset_preprocessing/ffhq/runme.py`,
   which aligns to 1500², then crops and resizes to 512 using EG3D's released
   `cropping_params.json`.

This produces `final_crops/*.png` (512×512 RGB) and `final_crops/dataset.json`
(25-dim camera labels, 139,914 entries including mirrors).

**GAIA uses EG3D's crop unmodified** — no crop, rescale, or resize parameter was changed.
EG3D's camera labels are only valid for that exact crop, so do not substitute your own
alignment.

Notes:
- You get **69,957** images, not 70,000 — 43 ids are absent from EG3D's `cropping_params.json`.
- Crops are reproducible but **not guaranteed bit-identical** — the final 512×512 resize
  quantises to uint8, and a value sitting on a rounding boundary can tie-break differently
  under another Pillow build, image-library version or CPU.

  Measured over a full 70,000-image reproduction against GAIA's released set: **139,812 of
  139,848 entries (99.974%) are bit-identical**. The remaining 18 source images (× 2 for
  mirrors) differ by **≤ 2/255 on 1–251 pixels of 262,144**, with alpha differing by ≤ 3/255
  where MODNet's matte is soft, and FLAME parameters shifted by ≤ 0.017. Camera parameters
  are always identical — they come from EG3D unchanged. The differences are well below
  training noise.

## 3. Prerequisites for Stages 2–6

| What | License | Where | How |
|---|---|---|---|
| MODNet code | Apache-2.0 | `gaia/third_party/MODNet/` | `git clone https://github.com/ZHKKKe/MODNet gaia/third_party/MODNet` |
| MODNet `webcam_portrait_matting` ckpt | Apache-2.0 | `gaia/third_party/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt` | [Google Drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR) |
| SMIRK code | MIT | `gaia/third_party/smirk/` | already a submodule — `git submodule update --init` |
| SMIRK `SMIRK_em1.pt` + `face_landmarker.task` | per repo | `gaia/third_party/smirk/{pretrained_models,assets}/` | the submodule's `quick_install.sh` |
| FLAME 2020 | research, account-gated | `gaia/flame/assets/FLAME2020/` | see `gaia/flame/assets/NOTE.md` |
| FLAME 2020, again for SMIRK | research, account-gated | `gaia/third_party/smirk/assets/FLAME2020/` | symlink the directory above: `ln -s "$PWD/gaia/flame/assets/FLAME2020" gaia/third_party/smirk/assets/FLAME2020` |

Stage 3 needs FLAME 2020 in **both** places. SMIRK's `src/FLAME/FLAME.py` hardcodes
`assets/FLAME2020/generic_model.pkl` (and four sibling `assets/...` paths) relative to its own
repo root, so the copy under `gaia/flame/` is not visible to it. Downloading once and symlinking
is enough — the FLAME model data is never redistributed by this repo.

Four pip extras beyond `environment.yml` (no new conda env, no new copyleft):

```bash
pip install mediapipe albumentations scikit-image timm
```

## 4. Stages 2–6

```python
from pathlib import Path
from dataset_preprocessing import (
    run_modnet, run_smirk, mirror_dataset, combine_to_dataset_json, pack_zip,
)

final_crops = Path('final_crops')          # from Stage 1
work        = Path('preprocessed')
smirk_out   = Path('smirk_flame')

# Stages 2 and 3 are independent -- both read final_crops/.
run_modnet(in_folder=final_crops, out_folder=work,
           num_workers=8, gpu_ids=list(range(8)))
run_smirk(in_folder=final_crops, out_folder=smirk_out,
          num_workers=8, gpu_ids=list(range(8)))

# Stage 4: mirror images + flip_yaw the camera poses.
import shutil
shutil.copy(final_crops / 'dataset.json', work / 'dataset.json')
mirror_dataset(source_dir=work)

# Stage 5: merge SMIRK FLAME params -> 380-dim labels (in place).
combine_to_dataset_json(source_dataset_json=work / 'dataset.json',
                        smirk_dir=smirk_out,
                        dest_dataset_json=work / 'dataset.json')

# Stage 6: pack.
pack_zip(source_dir=work, dest_zip=Path('FFHQ_512.zip'), resolution=(512, 512))
```

Run from the repo root so `import dataset_preprocessing` resolves.

## 5. Driving sequences for animation

`gaia/gen_animation.py` is driven by a `--flame-params` json: per-frame
SMIRK FLAME parameters (`pose_params`, `cam`, `shape_params`, `expression_params`,
`eyelid_params`, `jaw_params`) from a video of a face. `make_flame_params.py` makes one from a
folder of frames, using the same SMIRK setup as Stage 3. It needs the same prerequisites
(section 3), except FLAME 2020, which it does not read:

```bash
python dataset_preprocessing/make_flame_params.py \
    --frames path/to/frames --out gaia/eval/flame_params/driving.json --count 500
```

- Frames are `*.png` / `*.jpg` in one folder, taken in filename order (numerically for names
  like `0.png .. 1000.png`). `--start` / `--count` select a range. Any resolution works; the face
  crop comes from MediaPipe landmarks, exactly as in SMIRK's `demo.py --crop`.
- The parameters are the raw per-frame encoder outputs: no smoothing, no averaging.
- If MediaPipe finds no face in any frame, the script stops without writing, because a dropped
  frame would shift every later one.
- The source video's own license and consent terms are yours to check.

## 6. License

This directory is covered by the repo's `LICENSE`; third-party components are listed in
`THIRD-PARTY.txt`. Two obligations are yours, not ours: FFHQ is **CC BY-NC-SA 4.0** (and its
images carry their own per-image licenses), and FLAME 2020 requires accepting the Max Planck
license. MODNet is Apache-2.0, code and models alike.
