import os
import time
import uuid
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error


REFERENCE_IMAGES_PREFIX = "reference-images/"


class MinioStorage:
    """
    MinIO storage client for managing reference images.
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

    def upload_reference_image(self, image_bytes: bytes, filename: str, content_type: str = "image/png") -> Optional[str]:
        """
        Upload a reference image to MinIO.

        Args:
            image_bytes: Image data
            filename: Original filename (used for extension)
            content_type: MIME type of the image

        Returns:
            Public URL to the uploaded image, or None if upload failed
        """
        if not self._ensure_initialized():
            return None

        # Generate unique filename preserving extension
        ext = os.path.splitext(filename)[1] if "." in filename else ".png"
        image_id = str(uuid.uuid4())
        object_name = f"{REFERENCE_IMAGES_PREFIX}{image_id}{ext}"

        try:
            self.client.put_object(
                self.bucket,
                object_name,
                BytesIO(image_bytes),
                length=len(image_bytes),
                content_type=content_type,
            )

            url = f"{self.public_url}/{self.bucket}/{object_name}"
            print(f"[storage] Uploaded reference image: {url}")
            return url
        except Exception as e:
            print(f"[storage] Failed to upload reference image: {e}")
            return None

    def list_reference_images(self) -> list[dict]:
        """
        List all reference images in the bucket.

        Returns:
            List of dicts with 'name' and 'url' for each image
        """
        if not self._ensure_initialized():
            return []

        try:
            objects = self.client.list_objects(
                self.bucket,
                prefix=REFERENCE_IMAGES_PREFIX,
                recursive=True,
            )

            images = []
            for obj in objects:
                if obj.object_name and not obj.object_name.endswith("/"):
                    # Extract just the filename part
                    name = obj.object_name.replace(REFERENCE_IMAGES_PREFIX, "")
                    url = f"{self.public_url}/{self.bucket}/{obj.object_name}"
                    images.append({
                        "name": name,
                        "url": url,
                        "size": obj.size,
                        "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    })

            return images
        except Exception as e:
            print(f"[storage] Failed to list reference images: {e}")
            return []

    def get_random_reference_image(self) -> Optional[str]:
        """
        Get a random reference image URL.

        Returns:
            URL of a random reference image, or None if none available
        """
        import random

        images = self.list_reference_images()
        if not images:
            return None

        return random.choice(images)["url"]


# Global storage instance
storage = MinioStorage()
