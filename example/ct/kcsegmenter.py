import os
import torch.nn.functional as F
import torch
import numpy as np 
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from model.esfp import ESFPNetStructure

# Clear GPU cache
torch.cuda.empty_cache()
device = torch.device("cuda")
print("CUDA device: {}".format(device))
if not torch.cuda.is_available():
    print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")


def make_transform():
    train_transforms = transforms.Compose([
        transforms.Resize(size=(352, 352)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30)])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(352, 352)),
    ])

    return train_transforms, test_transforms

def _normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)

    return image

def convert2rgb(img):
    # image_stack = np.dstack((image_raw, image_raw, image_raw))
    stacked_img = np.stack((img,)*3, axis=-1)
    image_rgb = (stacked_img * 255).astype(np.uint8)
    return image_rgb

def load_model(file_path):
    feature_extractor = ESFPNetStructure()
    feature_extractor.load_state_dict(file_path)#, map_location = 'cuda')
    feature_extractor = feature_extractor.to(device)
    return feature_extractor

def preprocess_model(image_npy):
    image_norm = _normalize_image(image_npy)
    image_rgb = convert2rgb(image_norm)
    image = torch.from_numpy(image_rgb).to(torch.float)
    # image = image.unsqueeze(0)
    image = image.permute(2,0,1)
    image_transform = test_transform(image).to(device)
    return image_transform
    
def postprocess_model(preds):
    preds = F.upsample(preds, size=[512,512], mode='bilinear', align_corners=False)
    preds = preds.sigmoid()
    threshold = torch.tensor([0.5]).to(device)
    preds = (preds > threshold).float() * 1
    preds = preds.data.cpu().numpy().squeeze()
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    preds = torch.from_numpy(preds)
    return preds
        
    


if __name__ == '__main__':
    feature_extractor = load_model('/usr/work/kafka/ct/weight/ESFPNet_20240511.pt')
    print('done')
    input()
    feature_extractor.eval()
    train_transform, test_transform = make_transform()
    #DATASET
    data_path = '/data/ngocpt/Lang/09042024_670_Nhombenh/Data_infer/*/*/*/Image/*'

    for filename in tqdm(glob.glob(data_path)):

        mask_path = filename.replace('Image',"Mask")
        mask_path = mask_path.replace('image','mask')
        
        image = np.load(filename)
        mask = np.load(mask_path)
        
        image_transform = preprocess_model(image)

        preds = feature_extractor(image_transform.unsqueeze(0))
        preds = postprocess_model(preds)
        
        image_name = filename.split('/')[-1]
        output_path = '/data/ngocpt/Lang/09042024_670_Nhombenh/Output_infer/'+ '/'.join(filename.split('/')[-5:-2])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        out = np.concatenate([image,preds,mask],axis=1)
        np.save(os.path.join(output_path,image_name),out)
        plt.imshow(out, cmap='gray')
        plt.savefig('output_predict.png')