from minio import Minio

from minio import Minio
from minio.error import S3Error

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("localhost:9010",
        access_key="gfr1aFEpJ1zjQhdT",
        secret_key="lu2irzoNWNAqnn2Fk65ovGXpcEoqwdeZ",
        secure=False
    )

    # The file to upload, change this path if needed
    source_file = "./text.txt"

    # The destination bucket and filename on the MinIO server
    bucket_name = "test"
    destination_file = "my-test-file.txt"

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)