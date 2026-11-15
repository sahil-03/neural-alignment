from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_MB = 1024**2
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
_DELETE_BATCH = 1000  # S3 DeleteObjects limit


class R2Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="R2_", env_file=".env", extra="ignore")

    account_id: str
    access_key_id: SecretStr
    secret_access_key: SecretStr

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class ObjectInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    size: int
    etag: str
    last_modified: datetime


def _is_not_found(err: ClientError) -> bool:
    return str(err.response.get("Error", {}).get("Code")) in _NOT_FOUND_CODES


class R2Client:
    """Bucket-scoped R2 operations.

    Uploads go through boto3's managed transfer: large payloads are sent as multipart uploads,
    which only become visible once complete, so readers never see a partially written object.
    Missing keys raise FileNotFoundError, matching local-filesystem semantics.
    """

    def __init__(self, bucket: str, config: R2Config | None = None, *, max_concurrency: int = 16):
        self.bucket = bucket
        self.config = config or R2Config()  # type: ignore[call-arg]  # populated from env
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id.get_secret_value(),
            aws_secret_access_key=self.config.secret_access_key.get_secret_value(),
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max_concurrency,
            ),
        )
        self._transfer = TransferConfig(
            multipart_threshold=64 * _MB,
            multipart_chunksize=64 * _MB,
            max_concurrency=max_concurrency,
        )

    def __repr__(self) -> str:
        return f"R2Client(bucket={self.bucket!r}, account={self.config.account_id!r})"

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.upload_fileobj(io.BytesIO(data), self.bucket, key, Config=self._transfer)

    def get_bytes(self, key: str) -> bytes:
        buf = io.BytesIO()
        try:
            self._s3.download_fileobj(self.bucket, key, buf, Config=self._transfer)
        except ClientError as err:
            if _is_not_found(err):
                raise FileNotFoundError(f"r2://{self.bucket}/{key}") from err
            raise
        return buf.getvalue()

    def upload_file(self, path: str | Path, key: str) -> None:
        self._s3.upload_file(str(path), self.bucket, key, Config=self._transfer)

    def download_file(self, key: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self.bucket, key, str(path), Config=self._transfer)
        except ClientError as err:
            if _is_not_found(err):
                raise FileNotFoundError(f"r2://{self.bucket}/{key}") from err
            raise
        return path

    def head(self, key: str) -> ObjectInfo | None:
        try:
            resp = self._s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as err:
            if _is_not_found(err):
                return None
            raise
        return ObjectInfo(
            key=key,
            size=resp["ContentLength"],
            etag=resp["ETag"].strip('"'),
            last_modified=resp["LastModified"],
        )

    def exists(self, key: str) -> bool:
        return self.head(key) is not None

    def list(self, prefix: str = "") -> Iterator[ObjectInfo]:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield ObjectInfo(
                    key=obj["Key"],
                    size=obj["Size"],
                    etag=obj["ETag"].strip('"'),
                    last_modified=obj["LastModified"],
                )

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under `prefix`. Returns the number deleted."""
        if not prefix:
            raise ValueError("refusing to delete an entire bucket; pass a non-empty prefix")
        keys = [obj.key for obj in self.list(prefix)]
        for i in range(0, len(keys), _DELETE_BATCH):
            batch = keys[i : i + _DELETE_BATCH]
            self._s3.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
        return len(keys)
