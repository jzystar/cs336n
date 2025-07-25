import pickle
import os
from pathlib import Path
from typing import BinaryIO


def get_file_path(file_name):
    """
    generate file path for the dataset

    Args:
        file_name (str): name of the dataset

    Returns:
        str: file path of the dataset
    """
    data_dir = Path(__file__).parent.parent / 'data'
    file_path = os.path.join(data_dir, file_name)
    print("Dataset file path: ",file_path)
    return file_path


def save_file(data, path):
    """
    save the data to a file

    Args:
        data (Any): data to save
        path (str): path to save the data
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_file(path):
    """
    load the data from a file

    Args:
        path (str): path to load the data

    Returns:
        Any: data loaded from the file
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))