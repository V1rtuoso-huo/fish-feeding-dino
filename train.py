import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
class MaskDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None):
        self.img_dataset = datasets.ImageFolder(img_root, transform=transform)
        self.mask_root = mask_root
        self.mask_paths = []
        for path, _ in self.img_dataset.samples:
            rel_path = os.path.relpath(path, img_root)
            mask_name = os.path.splitext(rel_path)[0] + '.npy'
            mask_path = os.path.join(mask_root, mask_name)
            self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.img_dataset)
    
    def __getitem__(self, idx):
        img, label = self.img_dataset[idx]
        mask_path = self.mask_paths[idx]
        try:
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).float()
        except:
            mask = torch.ones_like(img[0:1])
        return img, mask, label
class MaskGuidedResNet(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        import torchvision
        base = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        orig_conv = base.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=orig_conv.kernel_size,
                               stride=orig_conv.stride, padding=orig_conv.padding,
                               bias=orig_conv.bias is not None)
        with torch.no_grad():
            self.conv1.weight[:, :3] = orig_conv.weight
            self.conv1.weight[:, 3:] = 0
            if self.conv1.bias is not None:
                self.conv1.bias = orig_conv.bias
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base.fc.in_features, num_classes)
    
    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
data_root = './dataset'
batch_size = 16
num_epochs = 100
lr = 1e-4
weight_decay = 1e-4
num_workers = 0
device = torch.device('cpu')
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MaskDataset(f'{data_root}/train', f'{data_root}/masks_train', train_transforms)
val_dataset   = MaskDataset(f'{data_root}/val',   f'{data_root}/masks_val',   val_test_transforms)
test_dataset  = MaskDataset(f'{data_root}/test',  f'{data_root}/masks_test',  val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
def train_one_model(seed, save_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = MaskGuidedResNet(num_classes=3, dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    counter = 0
    patience = 10
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for img, mask, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} train'):
            img, mask, labels = img.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        train_loss = running_loss / dataset_sizes['train']
        train_acc = running_corrects.double() / dataset_sizes['train']

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for img, mask, labels in val_loader:
                img, mask, labels = img.to(device), mask.to(device), labels.to(device)
                outputs = model(img, mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * img.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / dataset_sizes['val']
        val_acc = val_corrects.double() / dataset_sizes['val']

        print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f'验证Loss {best_val_loss:.4f}')
        else:
            counter += 1
            if counter >= patience:
                break

        scheduler.step()
    model.load_state_dict(best_model_wts)
    return model, history
seeds = [42, 123, 456] 
checkpoint_dir = 'checkpoints/ensemble'
os.makedirs(checkpoint_dir, exist_ok=True)

models = []
histories = []

for seed in seeds:
    save_path = os.path.join(checkpoint_dir, f'model_seed{seed}.pth')
    model, hist = train_one_model(seed, save_path)
    models.append(model)
    histories.append(hist)
test_corrects = 0
with torch.no_grad():
    for img, mask, labels in tqdm(test_loader, desc='Testing Ensemble'):
        img = img.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        avg_probs = 0
        for model in models:
            model.eval()
            outputs = model(img, mask)
            probs = torch.softmax(outputs, dim=1)
            avg_probs += probs
        avg_probs /= len(models)
        _, preds = torch.max(avg_probs, 1)
        test_corrects += torch.sum(preds == labels.data)

test_acc = test_corrects.double() / dataset_sizes['test']
print(f'集成模型测试集准确率: {test_acc:.4f}')
with open('results/ensemble_test_result.txt', 'w') as f:
    f.write(f'Ensemble Test Accuracy: {test_acc:.4f}\n')
    f.write(f'Seeds: {seeds}\n')
for i, (seed, hist) in enumerate(zip(seeds, histories)):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist['train_loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Val Loss')
    plt.legend(); plt.grid(True); plt.title(f'Seed {seed} Loss')
    plt.subplot(1,2,2)
    plt.plot(hist['val_acc'], label='Val Acc')
    plt.legend(); plt.grid(True); plt.title(f'Seed {seed} Val Acc')
    plt.savefig(f'results/ensemble_curve_seed{seed}.png')
    plt.close()

