import torch
import json
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Load the data from the JSON file
with open('/home/cxnam/Documents/MyWorkingSpace/Trainer/assets/samples/M003/samples_lm_vae/tensor_data.json', 'r') as json_file:
    data = json.load(json_file)

# Convert lists back to tensors
gt_img_feature = torch.tensor(data["gt_img_feature"])
img_feature = torch.tensor(data["pred_img_feature"])
lm_paths = data["lm_paths"]

# Print shapes to verify
print("gt_img_feature shape:", gt_img_feature.shape)
print("pred_img_feature shape:", img_feature.shape)

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

def decode(vae, latents):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        reconstructed = vae.decode(latents).sample
        reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    return reconstructed

pred_features = img_feature.detach().cpu()

inv_normalize = T.Compose([
    T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
    T.ToPILImage()
]) 

with torch.no_grad():
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

    for i in tqdm(range(pred_features.shape[0])):
        pf = pred_features[i].unsqueeze(0)
        # pf[:, :, 2*8:3*8, 1*8:3*8] = 0

        samples = vae.decode(pf)
        output = samples.sample[0]
        inv_image = inv_normalize(output)
        inv_image.save(f'./assets/samples/M003/samples_lm_vae/pred_{i:05d}.jpg')
        
        gt_img_path = lm_paths[i].replace("image_features", "images").replace("json", "jpg")
        gt_image = Image.open(gt_img_path)
        gt_image.save(f'./assets/samples/M003/samples_lm_vae/gt_{i:05d}.jpg')

        
        # break