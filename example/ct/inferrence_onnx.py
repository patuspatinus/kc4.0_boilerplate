import os
import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
import numpy as np 
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import onnxruntime

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
    
# Clear GPU cache
torch.cuda.empty_cache()
device = torch.device("cuda")
print("CUDA device: {}".format(device))
if not torch.cuda.is_available():
    print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

ort_session = onnxruntime.InferenceSession('weight/ESFPNet_20240511.onnx')

# feature_extractor.eval()
# feature_extractor.freeze()
train_transform, test_transform = make_transform()
#DATASET
data_path = 'data_tests/image_*.npy'
for filename in tqdm(glob.glob(data_path)):
    # print(filename)
    # filename = '/data/ngocpt/Lang/09042024_670_Nhombenh/Data_infer/AC1377/BU0295/CTB-128/Image/image_KC40_CT_0197_slide0180.npy'
    image_raw = np.load(filename)
    mask_path = filename.replace('Image',"Mask")
    mask_path = mask_path.replace('image','mask')
    mask = np.load(mask_path)
    
    image_norm = _normalize_image(image_raw)
    image_rgb = convert2rgb(image_norm)
    image = torch.from_numpy(image_rgb).to(torch.float)
    # image = image.unsqueeze(0)
    image = image.permute(2,0,1)
    image_transform = test_transform(image).to(device)
    # print(image_transform.shape)
    # preds = feature_extractor(image_transform.unsqueeze(0))
    input_name = ort_session.get_inputs()[0].name
    # x = torch.randn(1, 3, 352, 352, requires_grad=True).detach().cpu().numpy()
    ort_inputs = {input_name: image_transform.unsqueeze(0).detach().cpu().numpy()}
    preds = ort_session.run(None, ort_inputs)[0]
    preds = torch.from_numpy(preds).to(torch.float).to(device)
    # preds = torch.stack(preds).to(device)
    # print(preds)
    preds = F.upsample(preds, size=[512,512], mode='bilinear', align_corners=False)
    preds = preds.sigmoid()
    print(torch.min(preds), torch.max(preds))
    threshold = torch.tensor([0.5]).to(device)
    preds = (preds > threshold).float() * 1
    preds = preds.data.cpu().numpy().squeeze()
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    preds = torch.from_numpy(preds)
    # print(preds)
    output_path = 'data_tests/outputs/'+ '/'.join(filename.split('/')[-5:-2])
    # print(output_path)
    image_name = filename.split('/')[-1]
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    out = np.concatenate([image_raw,preds,mask],axis=1)
    np.save(os.path.join(output_path,image_name),out)
    plt.imshow(out, cmap='gray')
    plt.savefig(os.path.join(output_path,'output_predict.png'))