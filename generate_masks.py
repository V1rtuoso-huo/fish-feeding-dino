"""
DINO Soft Mask Generation Script
Pre-computes DINO ViT-B/8 self-attention masks for all dataset images.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# ========== Configuration ==========
data_root = './dataset'
output_dir = './dataset/masks'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Load DINO ViT-B/8 ==========
print('Loading DINO ViT-B/8...')
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
dino = dino.to(device)
dino.eval()

# ========== Image preprocessing for DINO ==========
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_mask(img_tensor):
    """Generate DINO self-attention soft mask for a single image.

    Args:
        img_tensor: (3, 224, 224) normalized tensor
    Returns:
        mask: (H, W) numpy array in [0, 1]
    """
    # Invert normalization to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_unnorm = img_tensor * std + mean
    img_unnorm = img_unnorm.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get self-attention from last block
        attentions = dino.get_last_selfattention(img_unnorm)  # (1, N+1, N+1)

        # [CLS] token attention to image patches
        cls_attn = attentions[0, 0, 1:]  # (N,)

        # Reshape to 2D
        nh = nw = int(np.sqrt(cls_attn.shape[0]))
        mask_2d = cls_attn.reshape(nh, nw)  # (14, 14)

        # Bilinear upsample to 224x224
        mask_2d = mask_2d.unsqueeze(0).unsqueeze(0)
        mask_up = F.interpolate(mask_2d, size=(224, 224), mode='bilinear', align_corners=False)
        mask_up = mask_up.squeeze().cpu().numpy()

        # Min-max normalize to [0, 1]
        mask_min, mask_max = mask_up.min(), mask_up.max()
        if mask_max - mask_min > 1e-8:
            mask_up = (mask_up - mask_min) / (mask_max - mask_min)

    return mask_up


# ========== Generate masks for all splits ==========
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(data_root, split)
    mask_dir = os.path.join(data_root, f'masks_{split}')
    os.makedirs(mask_dir, exist_ok=True)

    dataset = datasets.ImageFolder(img_dir, transform=transform)

    print(f'\nGenerating masks for {split} ({len(dataset)} images)...')

    for idx in tqdm(range(len(dataset))):
        img_tensor, _ = dataset[idx]
        mask = generate_mask(img_tensor)

        # Save mask with same relative path structure as images
        img_path = dataset.samples[idx][0]
        rel_path = os.path.relpath(img_path, img_dir)
        mask_name = os.path.splitext(rel_path)[0] + '.npy'

        # Ensure subdirectory exists
        mask_subdir = os.path.dirname(os.path.join(mask_dir, mask_name))
        os.makedirs(mask_subdir, exist_ok=True)

        np.save(os.path.join(mask_dir, mask_name), mask)

print('\nAll masks generated!')
