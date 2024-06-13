from minio import Minio

from minio import Minio
from minio.error import S3Error
import os

def fget_minio(image_url):
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("minio1:9000",
        access_key="test123",
        secret_key="05112004pd",
        secure=False
    )

    directory, filename = image_url.split('/', 1)
    print(directory)
    print(filename)

    # Name of the bucket containing the file
    bucket_name = directory

    # Name of the file you want to fetch
    file_name = '/' + filename

    # Local file path where the fetched file will be saved
    local_file_path = "/usr/work/kafka/ct/data_imgs" + file_name


    client.fget_object(bucket_name, file_name, local_file_path)
    print("File fetched successfully.")

    return local_file_path, file_name

# if __name__ == "__main__":
#     try:
#         fget_minio("nspqtest/z4739379682596_88027888913510bdc542473ff9bc6897.jpg")
#     except S3Error as exc:
#         print("error occurred.", exc)