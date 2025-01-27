import torch
# from nspq_model.infer_joint_Lung_lesions_correlation_api import saveResult, test_dataset, ESFPNetStructure

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    print('Only CPU available.')

import json
from datetime import datetime
import random
import asyncio
from kafka.consumer import KafkaEventConsumer
from kafka.producer import KafkaEventProducer
import uuid
import os


consumer = KafkaEventConsumer(
    bootstrap_servers=["localhost:9092"],
    topics=["broncho_segment"],
    group="consumer_1"
)

producer = KafkaEventProducer(
    bootstrap_servers=["localhost:9092"],
    topic=["broncho_segment","broncho_segment_annots"]
)

import os
import torch.nn.functional as F
import torch
import numpy as np 
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

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
    feature_extractor = torch.load(file_path,map_location = 'cuda')
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
        
    

def saveResult(local_file_path,file_name):
    feature_extractor = load_model('/home/kc_kafka/kc4.0utp-boilerplate/example/ct/weight/ESFPNet_20240511.pt')
    feature_extractor.eval()
    train_transform, test_transform = make_transform()
    #DATASET
    data_path = local_file_path

    for filename in tqdm(glob.glob(data_path)):

        mask_path = filename.replace('Image',"Mask")
        mask_path = mask_path.replace('image','mask')
        
        image = np.load(filename)
        mask = np.load(mask_path)
        
        image_transform = preprocess_model(image)

        preds = feature_extractor(image_transform.unsqueeze(0))
        preds = postprocess_model(preds)
        
        image_name = filename.split('/')[-1]
        output_path = '/home/kc_kafka/kc4.0utp-boilerplate/example/ct/data_annots/'+ '/'.join(filename.split('/')[-5:-2])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        out = np.concatenate([image,preds,mask],axis=1)
        np.save(os.path.join(output_path,image_name),out)
        plt.imshow(out, cmap='gray')
        plt.savefig('output_predict.png')
    return os.path.join(output_path,image_name)


from minio import Minio
from minio import Minio
from minio.error import S3Error
import os

from minIO.fget import fget_minio
from minIO.fput import fput_minio

async def handle_message(message):
    offset = message.offset
    topic = message.topic
    partition = message.partition
    data = json.loads(message.value)
    print(offset, topic, partition, data)

    # Kéo ảnh từ minio về
    try:
        local_file_path, filename = fget_minio(data["image_url"])
    except S3Error as exc:
        print("error occurred.", exc)

    # Xử lý ảnh
    annots_file_path = saveResult(local_file_path, filename)

    # Đẩy ảnh lên minio
    try:
        annots_minio_path = fput_minio(annots_file_path, filename)
    except S3Error as exc:
        print("error occurred.", exc)

    data_final = {
        "task_id": data["task_id"], 
        "service_type": "abc",
        "annotation": annots_minio_path
    }

    # Gửi lại event lên kafka
    await producer.flush(data_final, "broncho_segment_annots")

    await consumer.commit(topic=topic, partition=partition, offset=offset)

async def main():
    print('abc')
    # await producer.start()
    # data = {
    #     "task_id": str(uuid.uuid4()), 
    #     "image_type": "segmentation",
    #     "image_url": "nspqtest/ef9fe565-4bcc-40a1-ba77-2049d8585412.png",
    #     "time": datetime.now().isoformat()
    # }
    # await producer.flush(data, "broncho_segment")

    consumer.handle = handle_message
    await consumer.start()
    
    await consumer.stop()
    await producer.stop()

asyncio.run(main())