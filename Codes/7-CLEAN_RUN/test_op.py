import torch
from operateur.my_operators import InpaintingOperator

op = InpaintingOperator(img_size=4, n_channels=1, mask_type='box50', device='cpu')
print(f"mask:\n{op.mask.reshape(4,4)}")

x = torch.ones(16) * 0.5
y = op.create_observation(x, 0.0)
print(f"y:\n{y.reshape(4,4)}")
print(f"AtA_diag:\n{op.compute_AtA_diag().reshape(4,4)}")
print(f"Aty:\n{op.adjoint(y).reshape(4,4)}")