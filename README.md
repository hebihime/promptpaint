# promptpaint

type a sentence, get a wallpaper. this is the VQGAN+CLIP recipe from the
colab notebooks everyone is passing around this summer, untangled into a
plain script so i can run it in batches instead of babysitting a notebook.

## setup

    pip install -r requirements.txt
    bash download_models.sh   # ~1gb vqgan checkpoint, go make tea

CLIP and taming-transformers are not on pypi, they install straight from
github — that is what the git+ lines in requirements.txt are about.

## use

    python generate.py "a cozy cabin in a pine forest, matte painting"

images land in outputs/. on my 2060 a 480x480 run takes about ten
minutes for 300 iterations. on colab, clone the repo in a cell and run
the same command with ! in front — the free gpu is roughly twice as
fast, when it feels like showing up.

## batch mode

    python run_batch.py prompts.txt --iterations 400

one prompt per line, extra flags pass through to generate.py. every
image gets a .txt sidecar with the prompt, seed and settings, so a good
seed is never lost.

## remixing

    python generate.py "the same cabin but in autumn, matte painting" \
      --init-image outputs/a_cozy_cabin_in_a_pine_forest_matte_painting_seed7771.png

## prompt notes

- "matte painting" makes everything look like a video game loading screen (good)
- "watercolor" plus step size 0.05 is the reliable cozy combo
- seeds are in the filenames, 7771 has been lucky all autumn
- winter pack lives at the bottom of prompts.txt

## plan

- [x] batch mode: prompts.txt in, wallpapers out
- [x] init from an image (remix an old wallpaper)
- [ ] bigger sizes without running out of vram
