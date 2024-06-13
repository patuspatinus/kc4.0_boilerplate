import torch
from infer_joint_Lung_lesions_correlation_api import saveResult, test_dataset, ESFPNetStructure

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
    bootstrap_servers=["kafka0:29092"],
    topics=["nspq"],
    group="consumer_1"
)

producer = KafkaEventProducer(
    bootstrap_servers=["kafka0:29092"],
    topic=["nspq","nspq_annots"]
)


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
    await producer.flush(data_final, "nspq_annots")

    await consumer.commit(topic=topic, partition=partition, offset=offset)

async def main():
    await producer.start()

    consumer.handle = handle_message
    await consumer.start()
    
    await consumer.stop()
    await producer.stop()

asyncio.run(main())