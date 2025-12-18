import os
import time
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error


class MinioStorage:
    """
    MinIO storage client for uploading generated images.
    """

    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "game-images")
        self.public_url = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000").rstrip("/")

        self.client: Optional[Minio] = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazily initialize the MinIO client and bucket."""
        if self._initialized:
            return True

        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=False,
            )

            # Wait for MinIO to be available
            for attempt in range(10):
                try:
                    if not self.client.bucket_exists(self.bucket):
                        self.client.make_bucket(self.bucket)
                        print(f"[storage] Created bucket: {self.bucket}")

                    # Set bucket policy to allow public read access
                    policy = f'''{{
                        "Version": "2012-10-17",
                        "Statement": [
                            {{
                                "Effect": "Allow",
                                "Principal": {{"AWS": ["*"]}},
                                "Action": ["s3:GetObject"],
                                "Resource": ["arn:aws:s3:::{self.bucket}/*"]
                            }}
                        ]
                    }}'''
                    self.client.set_bucket_policy(self.bucket, policy)

                    self._initialized = True
                    print(f"[storage] MinIO initialized: {self.endpoint}/{self.bucket}")
                    return True
                except S3Error as e:
                    if attempt < 9:
                        print(f"[storage] Waiting for MinIO... ({e})")
                        time.sleep(2)
                    else:
                        raise
        except Exception as e:
            print(f"[storage] Failed to initialize MinIO: {e}")
            return False

        return False

    def upload_image(self, image_bytes: bytes, game_id: str, round_id: str, image_id: str) -> Optional[str]:
        """
        Upload an image to MinIO and return its public URL.

        Args:
            image_bytes: PNG image data
            game_id: Game ID for path organization
            round_id: Round ID for path organization
            image_id: Unique image ID

        Returns:
            Public URL to the uploaded image, or None if upload failed
        """
        if not self._ensure_initialized():
            return None

        object_name = f"{game_id}/{round_id}/{image_id}.png"

        try:
            self.client.put_object(
                self.bucket,
                object_name,
                BytesIO(image_bytes),
                length=len(image_bytes),
                content_type="image/png",
            )

            url = f"{self.public_url}/{self.bucket}/{object_name}"
            print(f"[storage] Uploaded image: {url}")
            return url
        except Exception as e:
            print(f"[storage] Failed to upload image: {e}")
            return None


# Global storage instance
storage = MinioStorage()
