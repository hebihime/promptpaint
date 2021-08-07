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

## plan

- [ ] batch mode: prompts.txt in, wallpapers out
- [ ] init from an image (remix an old wallpaper)
- [ ] bigger sizes without running out of vram
