"""text to wallpaper with VQGAN+CLIP.

the recipe from the colab notebooks everyone is passing around this
summer, untangled into a plain script. VQGAN draws, CLIP judges, Adam
nudges the latent until the picture matches the prompt.

run download_models.sh first, then:
    python generate.py "a cozy cabin in a pine forest, matte painting"
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import clip
from taming.models import vqgan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_NORMALIZE = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


class MakeCutouts(torch.nn.Module):
    """CLIP looks at 224x224, so it judges a pile of random crops."""

    def __init__(self, cut_size, num_cuts):
        super().__init__()
        self.cut_size = cut_size
        self.num_cuts = num_cuts

    def forward(self, img):
        _, _, h, w = img.shape
        biggest = min(h, w)
        cuts = []
        for _ in range(self.num_cuts):
            size = int(torch.rand([]) * (biggest - self.cut_size)) + self.cut_size
            x = int(torch.randint(0, w - size + 1, ()))
            y = int(torch.randint(0, h - size + 1, ()))
            crop = img[:, :, y:y + size, x:x + size]
            cuts.append(F.interpolate(crop, self.cut_size, mode="bilinear",
                                      align_corners=False))
        return torch.cat(cuts)


def slugify(prompt):
    slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:48]


def load_vqgan(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = vqgan.VQModel(**config.model.params)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False)
    return model.to(DEVICE)


def random_latent(model, width, height):
    factor = 2 ** (model.decoder.num_resolutions - 1)
    tokens_x, tokens_y = width // factor, height // factor
    codebook = model.quantize.embedding.weight
    indices = torch.randint(codebook.shape[0], [tokens_y * tokens_x], device=DEVICE)
    z = F.one_hot(indices, codebook.shape[0]).float() @ codebook
    return z.view([1, tokens_y, tokens_x, -1]).permute(0, 3, 1, 2)


def quantize(z, codebook):
    # snap each latent vector to its nearest codebook entry on the way
    # forward, but keep the gradient as if we never snapped
    flat = z.movedim(1, 3)
    dists = (flat.pow(2).sum(dim=-1, keepdim=True)
             - 2 * flat @ codebook.T
             + codebook.pow(2).sum(dim=1))
    snapped = codebook[dists.argmin(-1)]
    return (flat + (snapped - flat).detach()).movedim(3, 1)


def synth(model, z):
    z_q = quantize(z, model.quantize.embedding.weight)
    return model.decode(z_q).add(1).div(2).clamp(0, 1)


def spherical_loss(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a.sub(b).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()


def generate(prompt, args, out_dir):
    model = load_vqgan(args.config, args.checkpoint)
    perceptor, _ = clip.load("ViT-B/32", device=DEVICE)
    perceptor.eval()

    text_embed = perceptor.encode_text(clip.tokenize(prompt).to(DEVICE)).detach()
    make_cutouts = MakeCutouts(perceptor.visual.input_resolution, args.num_cuts)

    seed = args.seed if args.seed is not None else int(torch.randint(0, 100000, ()))
    torch.manual_seed(seed)
    print(f"seed {seed}")

    z = random_latent(model, args.size, args.size)
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=args.step_size)

    for i in range(args.iterations + 1):
        optimizer.zero_grad()
        image = synth(model, z)
        embeds = perceptor.encode_image(CLIP_NORMALIZE(make_cutouts(image)))
        loss = spherical_loss(embeds, text_embed)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            print(f"iteration {i}: loss {loss.item():.4f}")

    stem = f"{slugify(prompt)}_seed{seed}"
    out_path = out_dir / f"{stem}.png"
    TF.to_pil_image(synth(model, z)[0].detach().cpu()).save(out_path)
    (out_dir / f"{stem}.txt").write_text(
        f"prompt: {prompt}\n"
        f"seed: {seed}\n"
        f"iterations: {args.iterations}\n"
        f"size: {args.size}x{args.size}\n"
        f"step size: {args.step_size}\n"
    )
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="text to wallpaper with VQGAN+CLIP")
    parser.add_argument("prompt")
    parser.add_argument("--size", type=int, default=384,
                        help="square edge. 480 wants more vram than my 2060 has")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--num-cuts", type=int, default=32)
    parser.add_argument("--seed", type=int, help="reuse a good seed")
    parser.add_argument("--config", default="checkpoints/vqgan_imagenet_f16_16384.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/vqgan_imagenet_f16_16384.ckpt")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    generate(args.prompt, args, out_dir)


if __name__ == "__main__":
    main()
