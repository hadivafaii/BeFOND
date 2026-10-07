"""Small file utilities shared by mechinterp applications."""

import hashlib
import pathlib

from tqdm import tqdm


def sha256_file(
		path: str | pathlib.Path,
		*,
		show_progress: bool = False,
		progress_description: str = "Hashing file",
		chunk_size: int = 8 * 1024 * 1024,
		) -> str:
	"""Return a file's SHA-256 digest without loading it into memory."""
	path = pathlib.Path(path)
	digest = hashlib.sha256()
	progress = tqdm(
		total=path.stat().st_size,
		desc=progress_description,
		unit="B",
		unit_scale=True,
		unit_divisor=1024,
		disable=None if show_progress else True,
	)
	with open(path, "rb") as file, progress:
		while chunk := file.read(chunk_size):
			digest.update(chunk)
			progress.update(len(chunk))
	return digest.hexdigest()


__all__ = ["sha256_file"]
