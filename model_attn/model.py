import torch
from torch import nn

from typing import Union

from model_attn.audio_encoder import AudioEncoder
from model_attn.visual_encoder import VisualEncoder

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import torchvision

class Model(nn.Module):

    def __init__(self,
                 img_dim: int,
                 lm_dim: int,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        self.img_dim = img_dim
        self.lm_dim = lm_dim
        
        self.audio = AudioEncoder()
        self.visual = VisualEncoder(num_classes=8)
    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio(audio.to(device=self.device, dtype=torch.float32))
        return audio_embedding
    
    def encode_visual(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.visual((images.to(device=self.device, dtype=torch.float32), context))
    
    def forward(self,
                audio,
                visual,
                emotion,
                grad_map = False,
                save_folder = ""
                ):
        audio_features = self.encode_audio(audio)
        out_features = self.encode_visual(images=visual, context=audio_features)
        
        loss = self.loss_fn(out_features, emotion)
        

        #         attn_map = x1.clone()
        #         attn_map = torch.sigmoid(attn_map)
        #         attn_map = attn_map.detach().cpu().numpy()
        #         for i in range(attn_map.shape[0]):
        #             # heatmap = (attn_map[i][0] * 255).astype(np.uint8)
        #             heatmap = attn_map[i][0]
        #             heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255

        #             plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        #             plt.axis('off') 
        #             plt.savefig(f'{save_folder}/image_{i:05d}.jpg', bbox_inches='tight', pad_inches=0) 
        #             plt.close()  

        return (out_features), loss
        
    def loss_fn(self, pred_features, gt_features):
        gt_tensor_indices = torch.argmax(gt_features, dim=1)
        
        loss = nn.CrossEntropyLoss()(pred_features, gt_tensor_indices)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        phoneme, audio, visual, emotion = batch
        emotion = emotion.to(self.module.device)

        _, loss = self(
            audio = audio, 
            visual = visual,
            emotion = emotion
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            phoneme, audio, visual, emotion = batch
            emotion = emotion.to(self.module.device)
            
            (pred_feature), _ = self(
                audio = audio, 
                visual = visual,
                emotion = emotion
            )
            
            gt_tensor_indices = torch.argmax(emotion, dim=1)
            
        return {"y_pred": pred_feature, "y": emotion, "y_indices": gt_tensor_indices}
        
    def inference(self, batch, device, save_folder):
        phoneme, audio, visual, emotion = batch
        visual = visual.to(self.module.device)
        
        audio_features = self.module.encode_audio(audio)
        
        model = self.module.visual
        layer_name = [model.conv4[-1]]
        cam = GradCAM(model=model, target_layers=layer_name)
        targets = []
        for e in torch.argmax(emotion, dim=1):
            targets.append(ClassifierOutputTarget(e))
        grayscale_cam = cam(input_tensor=(visual, audio_features), targets=targets)
        
        
        self.inv_normalize = torchvision.transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5])
        visual = self.inv_normalize(visual)
        visual = visual.detach().cpu().numpy()
        visual = visual * 255.0

        for i in range(grayscale_cam.shape[0]):
            rgb_img = visual[i].transpose((1,2,0))
            gs_cam = grayscale_cam[i]
            # gs_cam = sigmoid(gs_cam, 50, 0.5, 1)
            # visualization = show_cam_on_image(rgb_img, gs_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.3)
            visualization = superimpose(rgb_img, gs_cam, 0.8, emphasize=True)
            cv2.imwrite(f'{save_folder}/image_{i:05d}.jpg', visualization)
        
def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 100, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = 0.8
    superimposed_img = heatmap * (1-hif) + img_bgr * hif
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_rgb