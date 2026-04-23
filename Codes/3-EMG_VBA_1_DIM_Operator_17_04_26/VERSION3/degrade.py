"""
  python degrade.py --image photo.png --sigma_blur 1.0
"""
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


def load_and_resize(path, img_size=64):
    img = Image.open(path).convert('L')
    img = img.resize((img_size, img_size), Image.BICUBIC)
    return np.array(img, dtype=np.float64) / 255.0


def blur(x, sigma):
    return gaussian_filter(x, sigma=sigma, mode='wrap')



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      type=str, required=True)
    p.add_argument("--sigma_blur", type=float, default=1.0)
    p.add_argument("--img_size",   type=int, default=64)
    p.add_argument("--output_dir", type=str, default='observations')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]

    # 1. Charger et resize
    x0 = load_and_resize(args.image, args.img_size)

    # 2. Flouter (sans bruit)
    y = blur(x0, args.sigma_blur)

    # 3. Passer en [-1, 1]
    x0_11 = 2.0 * x0 - 1.0
    y_11  = 2.0 * y  - 1.0

    # 4. Sauvegarder
    np.save(os.path.join(args.output_dir, f"{name}_clean.npy"), x0_11)
    np.save(os.path.join(args.output_dir, f"{name}_blur.npy"),  y_11)
    Image.fromarray((y * 255).clip(0, 255).astype(np.uint8), mode='L') \
         .save(os.path.join(args.output_dir, f"{name}_blur.png"))

    print(f"Image propre : {args.img_size}x{args.img_size}, sigma_blur = {args.sigma_blur}")
    print(f"  {args.output_dir}/{name}_clean.npy  ([-1,1])")
    print(f"  {args.output_dir}/{name}_blur.npy   ([-1,1])")
    print(f"  {args.output_dir}/{name}_blur.png")


if __name__ == "__main__":
    main()