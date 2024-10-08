import torch
from piq import ssim, psnr, FID
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
])

gt_list = []
pred_list = []
gt_image_paths = []
pred_image_paths = []
for i in range(30):
    # gt_path = f'/home/cxnam/Documents/MEAD/M003/images/front_happy_level_1/001/{(i+1):05d}.jpg'
    # pred_path = f'/home/cxnam/Documents/MyWorkingSpace/Trainer/inference/samples/lm2face/pred_image_{i:05d}.jpg'
    gt_path = f'/home/cxnam/Documents/MyWorkingSpace/Trainer/inference/fs-vid2vid/000_real/{(i+1):05d}.jpg'
    pred_path = f'/home/cxnam/Documents/MyWorkingSpace/Trainer/inference/fs-vid2vid/000/{(i+1):05d}.jpg'
    gt_image_paths.append(gt_path)
    pred_image_paths.append(pred_path)
    gt = Image.open(gt_path)
    pred = Image.open(pred_path)
    gt_list.append(transform(gt))
    pred_list.append(transform(pred))
    

gt_list = torch.stack(gt_list, dim=0)
pred_list = torch.stack(pred_list, dim=0)
ssim_score = ssim(gt_list, pred_list, data_range=1.)
print(ssim_score)

psnr_score = psnr(gt_list, pred_list, data_range=1.)
print(psnr_score)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
        
        if self.transform:
            image = self.transform(image)
            
        return {"images": image}

gt_dataset = ImageDataset(image_paths=gt_image_paths, transform=transform)
gt_dataloader = DataLoader(gt_dataset, batch_size=1, shuffle=False)

pred_dataset = ImageDataset(image_paths=pred_image_paths, transform=transform)
pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

fid_metric = FID()
gt_feats = fid_metric.compute_feats(gt_dataloader)
pred_feats = fid_metric.compute_feats(pred_dataloader)
fid = fid_metric(gt_feats, pred_feats)
print(fid)