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
import onnxruntime
import uuid
import os

from PIL import Image


consumer = KafkaEventConsumer(
    bootstrap_servers=["kafka0:29092"],
    topics=["ct"],
    group="consumer_1"
)

producer = KafkaEventProducer(
    bootstrap_servers=["kafka0:29092"],
    topic=["ct","ct_annots"]
)

import numpy
from scipy.ndimage import median_filter
from skimage import measure, morphology
from skimage.measure import label, regionprops, regionprops_table
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

# Preprocess func
def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            deltas[i][tuple(slicer)] = numpy.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            matrices[i][tuple(slicer)] = numpy.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))

    return out

def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = numpy.mean(img)
    std = numpy.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = numpy.mean(middle)  
    max = numpy.max(img)
    min = numpy.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(numpy.reshape(middle,[numpy.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = numpy.mean(centers)
    thresh_img = numpy.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,numpy.ones([4,4]))
    dilation = morphology.dilation(eroded,numpy.ones([10,10]))
    labels = measure.label(dilation)
    # label_vals = numpy.unique(labels)
    regions = regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = numpy.ndarray([512,512],dtype=numpy.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + numpy.where(labels==N,1,0)
    mask = morphology.dilation(mask,numpy.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*img

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
    # feature_extractor = torch.load(file_path,map_location = 'cuda')
    feature_extractor = onnxruntime.InferenceSession(file_path)
    # feature_extractor = feature_extractor.to(device)
    return feature_extractor

def preprocess_model(image_npy, test_transform):
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
        
    

def saveResult(local_file_path, file_name):
    # feature_extractor = load_model('/usr/work/kafka/ct/weight/ESFPNet_20240511.pt')
    feature_extractor = load_model('weight/ESFPNet_20240511.onnx')

    print("model")
    # feature_extractor.eval()
    train_transform, test_transform = make_transform()
    #DATASET
    data_path = local_file_path
    for filename in tqdm(glob.glob(data_path)):

        # mask_path = filename.replace('Image',"Mask")
        # mask_path = mask_path.replace('image','mask')
        
        if filename.endswith(".npy"):
            image = np.load(filename)
        elif filename.endswith(".png"):
            image = np.array(Image.open(filename))
        # image = segment_lung(image)

        # mask = np.load(mask_path)
        
        image_transform = preprocess_model(image, test_transform)

        input_name = feature_extractor.get_inputs()[0].name
        ort_inputs = {input_name: image_transform.unsqueeze(0).detach().cpu().numpy()}

        preds = feature_extractor.run(None, ort_inputs)[0]
        preds = torch.from_numpy(preds).to(torch.float).to(device)
        # preds = feature_extractor(image_transform.unsqueeze(0))
        preds = postprocess_model(preds)
        
        image_name = "annot_" + os.path.basename(filename).split('.')[0] + '.png'
        output_path = '/usr/work/kafka/ct/data_annots/'
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        # out = np.concatenate([image,preds,mask],axis=1)
        out = np.concatenate([image, preds], axis=1)
        # np.save(os.path.join(output_path,image_name),out)
        plt.imshow(out, cmap='gray')
        plt.savefig(os.path.join(output_path,image_name))
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
        # annots_minio_path = fput_minio("/usr/work/kafka/ct/data_imgs/440899195_2230133187327066_329869532347743069_n.png", "440899195_2230133187327066_329869532347743069_n.png")
    except S3Error as exc:
        print("error occurred.", exc)

    data_final = {
        "task_id": data["task_id"], 
        "annots_url": annots_minio_path,
        "time": data["time"],
        "patient_id": data["patient_id"], 
        "patient_birth_date": data["patient_birth_date"],
        "patient_sex": data["patient_sex"],
        'study_date': data["study_date"],
        'accession_number': data["accession_number"],
        'study_instance_uid': data["study_instance_uid"], 
        'study_id': data["study_id"],
        'requested_procedure_description': data['requested_procedure_description'],
        'instance_number': data['instance_number'],
        'body_part_examined': data['body_part_examined'],
        'modality': data['modality'],
        'sop_instance_uid': data['sop_instance_uid']
    }

    # Gửi lại event lên kafka
    await producer.flush(data_final, "ct_annots")

    await consumer.commit(topic=topic, partition=partition, offset=offset)

async def main():
    await producer.start()
    consumer.handle = handle_message
    await consumer.start()
    
    await consumer.stop()
    await producer.stop()

asyncio.run(main())