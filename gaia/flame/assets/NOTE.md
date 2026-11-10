# Assets

Put the FLAME 2020 model here; `setup_assets.sh` creates everything else. Please see `README.md` for details.

**FLAME 2020 is not downloaded by any script.** It is licensed by the Max Planck Institute and
gated behind a registration you complete yourself: accept the licence at
<https://flame.is.tue.mpg.de/>, download `FLAME2020.zip`, and unzip it here so that
`FLAME2020/generic_model.pkl` exists. `setup_assets.sh` refuses to run until it does.

Then run:
```
cd <GAIA_root>/gaia/flame/assets
bash setup_assets.sh
```

Expected file structure at `gaia/flame/assets/`:
```
- FLAME2020/                      <- you download this yourself (MPI licence)
    - female_model.pkl
    - generic_model.pkl
    - male_model.pkl
- l_eyelid.npy                    -> symlink into third_party/smirk/assets/
- r_eyelid.npy                    -> symlink into third_party/smirk/assets/
- landmark_embedding.npy          -> symlink into third_party/smirk/assets/
- mediapipe_landmark_embedding/   -> symlink into third_party/smirk/assets/
    - mediapipe_landmark_embedding.npz
    - readme
- head_template.obj               <- downloaded from FlashAvatar, pinned commit
- flame_uv_close_mouth.obj        <- generated
- flame_uv_masks/                 <- generated
    - uv_eye_mask_512.png
    - uv_eye_mask.png
    - uv_eye_region_mask_512.png
    - uv_eye_region_mask.png
    - uv_face_eye_teeth_mask_512.png
    - uv_face_eye_teeth_mask.png
    - uv_forehead_mask_512.png
    - uv_forehead_mask.png
    - uv_lips_mask_512.png
    - uv_lips_mask.png
- flame_to_canonical_params.json  <- tracked in git
```

The four `->` entries are symlinks created by `setup_assets.sh`; the files live in the SMIRK
submodule, pinned by gitlink, so `./install.sh submodules` must have run first. They resolve
transparently, and `verify_assets.py` hashes them through the link like any other file.

To check an existing install without downloading anything:
```
python verify_assets.py --numeric
```

`setup_assets.sh` runs `verify_assets.py` as its last step, so a normal install is verified
automatically; the command above is for re-checking later.