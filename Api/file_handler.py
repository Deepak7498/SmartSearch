import json
import numpy as np
import chardet
import mimetypes
import redis
from langchain_community.vectorstores.redis import Redis

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
VECTOR_DIM = 128
INDEX_NAME = "Smart_search"


def process_file(file):
    try:
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            raise ValueError("Unable to determine file type.")

        raw_data = file.read()

        if mime_type.startswith("text"):
            detected = chardet.detect(raw_data)
            encoding = detected.get("encoding", "utf-8")

            # Decode the text content
            content = raw_data.decode(encoding)
            if not content.strip():
                raise ValueError("File is empty.")
            return content

        elif mime_type == "application/json":
            # Decode JSON file content
            import json
            try:
                content = raw_data.decode("utf-8")
                json_data = json.loads(content)
                return json.dumps(json_data, indent=2)
            except Exception as e:
                raise ValueError(f"Invalid JSON file: {e}")

        elif mime_type.startswith("application"):
            return f"File is a binary type: {mime_type}"

        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def process_video_file(file):
    try:
        raise ValueError("File is empty.")

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def save_tags_in_redis(request):
    try:
        content_data = {
            "contentId": request.contentId,
            "contentTitle": request.contentTitle,
            "tags": request.tags
        }
        redis_key = f"content:{request.contentId}"
        redis_client.set(redis_key, json.dumps(content_data))
        return {"message": "Tags saved successfully.", "redis_key": redis_key}
    except Exception as e:
        raise ValueError(f"Error saving tags in Redis: {e}")

def save_task_in_redis_vectors(request):
    try:
        create_vector_index(INDEX_NAME, VECTOR_DIM)
        vector = np.random.rand(VECTOR_DIM)
        redis_key = save_vector(INDEX_NAME, vector, request.contentId, request.contentTitle, request.tags)
        return {"message": "Vector and metadata saved successfully under key.", "redis_key": redis_key}
    except Exception as e:
        raise ValueError(f"Error saving task in Redis: {e}")


def save_vector(index_name: str, vector: np.ndarray, contentId: int, contentTitle: str, tags: list[str]):
    try:
        redis_key = f"{index_name}:{contentId}"

        # Convert vector to bytes (float32 array)
        vector_bytes = vector.astype(np.float32).tobytes()

        # Save data to Redis as a hash
        redis_client.hset(
            redis_key,
            mapping={
                "contentId": contentId,
                "contentTitle": contentTitle,
                "tags": json.dumps(tags),
                "vector": vector_bytes
            }
        )
        return redis_key
    except Exception as e:
        return ValueError (f"Error saving vector: {e}")


def create_vector_index(index_name: str, vector_dim: int):
    """
    Create a vector index in Redis for storing vector data.
    """
    try:
        redis_client.execute_command(
            "FT.CREATE",
            index_name,
            "ON", "HASH",
            "PREFIX", "1", f"{index_name}:",
            "SCHEMA",
            "contentId", "NUMERIC",  # Metadata: content ID
            "contentTitle", "TEXT",  # Metadata: content title
            "tags", "TEXT",          # Metadata: tags
            "vector", "VECTOR", "FLAT",  # Vector field
            "6",                      # Vector index parameters
            "TYPE", "FLOAT32",
            "DIM", vector_dim,
            "DISTANCE_METRIC", "COSINE"
        )
        print(f"Vector index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")
