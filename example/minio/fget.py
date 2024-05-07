from minio import Minio

from minio import Minio
from minio.error import S3Error
import os

def fget_minio(image_url):
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("localhost:9010",
        access_key="60PYdF5SMtZHggt9",
        secret_key="cYjh2Rn3b6DL2QJFT0cjPYHhV22VRF1U",
        secure=False
    )

    directory, filename = os.path.split(image_url)

    # Name of the bucket containing the file
    bucket_name = directory

    # Name of the file you want to fetch
    file_name = filename

    # Local file path where the fetched file will be saved
    local_file_path = "/workspace/ailab/kc4.0utp-boilerplate/data_minio_test/" + file_name
    print(local_file_path)

    client.fget_object(bucket_name, file_name, local_file_path)
    print("File fetched successfully.")

if __name__ == "__main__":
    try:
        fget_minio("nspqtest/z4739379682596_88027888913510bdc542473ff9bc6897.jpg")
    except S3Error as exc:
        print("error occurred.", exc)