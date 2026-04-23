import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import UNet
from diffusion.Schedules import DDPMSchedule, VESchedule, VPOUSchedule
from diffusion.Training import train


# ==================
# HYPERPARAMÈTRES
# ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, '..', 'Data', 'ffhq256')
MODEL_NAME = 'unet_ffhq64'
IMG_SIZE = 64
IN_CH = 1
BATCH_SIZE = 32
EPOCHS = 1000
LR = 2e-4
EMA_DECAY = 0.9999
GRAD_CLIP = 1.0
DEVICE = 'mps'

# Schedule
SCHEDULE_TYPE  = 'ddpm'   # 'ddpm' | 've' | 'vpou'
T = 1000
BETA_START = 1e-4
BETA_END = 0.02

# UNet
BASE_CH = 128
TIME_DIM = 256
# ==================


class ImageFolderDataset(Dataset):
    """Charge toutes les images d'un dossier, les resize et normalise dans [-1, 1]."""
    EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

    def __init__(self, folder, img_size, in_ch=1):
        self.paths = []
        for ext in self.EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"Aucune image trouvée dans {folder}")
        self.img_size = img_size
        self.mode = 'L' if in_ch == 1 else 'RGB'
        self.in_ch = in_ch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert(self.mode)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.in_ch == 1:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        arr = 2.0 * arr - 1.0
        return torch.from_numpy(arr)


def build_schedule(schedule_type, T, beta_start, beta_end):
    if schedule_type == 'ddpm':
        return DDPMSchedule(T=T, beta_start=beta_start, beta_end=beta_end)
    if schedule_type == 've':
        return VESchedule(T=T)
    if schedule_type == 'vpou':
        return VPOUSchedule(T=T)
    raise ValueError(f"Schedule inconnu : {schedule_type}")


def train_model(
    train_dir,
    model_name,
    img_size=64,
    in_ch=1,
    batch_size=32,
    epochs=100,
    lr=2e-4,
    ema_decay=0.9999,
    grad_clip=1.0,
    device='mps',
    schedule_type='ddpm',
    T=1000,
    beta_start=1e-4,
    beta_end=0.02,
    base_ch=128,
    time_dim=256,
    num_workers=0,
    ckpt_root=None,
):
    """
    Entraîne un UNet de diffusion sur les images de `train_dir`.
    Les checkpoints sont sauvegardés dans `<ckpt_root>/<model_name>/`.

    Args :
        train_dir     : dossier contenant les images d'entraînement
        model_name    : nom du modèle (sert de sous-dossier checkpoints)
        img_size      : taille des images (carré)
        in_ch         : nb de canaux (1 = niveau de gris, 3 = RGB)
        batch_size    : taille de batch
        epochs        : nombre d'époques
        lr            : learning rate
        ema_decay     : décroissance EMA
        grad_clip     : clipping gradient
        device        : 'mps' | 'cuda' | 'cpu'
        schedule_type : 'ddpm' | 've' | 'vpou'
        T             : nombre de pas de diffusion
        beta_start    : beta min (DDPM)
        beta_end      : beta max (DDPM)
        base_ch       : nb canaux de base du UNet
        time_dim      : dim du time embedding
        num_workers   : workers DataLoader
        ckpt_root     : racine des checkpoints (défaut : BASE_DIR/checkpoints)

    Returns :
        net, ema_net, losses
    """
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    if ckpt_root is None:
        ckpt_root = os.path.join(BASE_DIR, 'checkpoints')
    ckpt_dir = os.path.join(ckpt_root, model_name)

    print(f"[Training] modèle={model_name} | données={train_dir}")
    print(f"[Training] img_size={img_size} in_ch={in_ch} batch={batch_size} epochs={epochs} device={device}")

    dataset = ImageFolderDataset(train_dir, img_size=img_size, in_ch=in_ch)
    print(f"[Training] {len(dataset)} images chargées")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device == 'cuda'),
    )

    net = UNet(in_ch=in_ch, base_ch=base_ch, time_dim=time_dim)
    schedule = build_schedule(schedule_type, T, beta_start, beta_end)

    net, ema_net, losses = train(
        net=net,
        schedule=schedule,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        ema_decay=ema_decay,
        grad_clip=grad_clip,
        ckpt_dir=ckpt_dir,
        device=device,
    )

    print(f"[Training] terminé — checkpoints dans {ckpt_dir}")
    return net, ema_net, losses


if __name__ == "__main__":
    train_model(
        train_dir=TRAIN_DIR,
        model_name=MODEL_NAME,
        img_size=IMG_SIZE,
        in_ch=IN_CH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        ema_decay=EMA_DECAY,
        grad_clip=GRAD_CLIP,
        device=DEVICE,
        schedule_type=SCHEDULE_TYPE,
        T=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    )
