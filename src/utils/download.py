import logging
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def download_file(
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    user_agent: str = "Mozilla/5.0",
) -> None:
    """Download a file with retries and atomic write.

    Args:
        url: The URL to download.
        dest_path: The local path to save the file to.
        max_retries: Maximum number of retries for transient errors.
        backoff_factor: Multiplier for sleep time between retries.
        user_agent: User-Agent header to use.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    last_error: Optional[Exception] = None
    
    # Configure opener with User-Agent
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", user_agent)]
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Downloading {url} (attempt {attempt}/{max_retries})")
            with opener.open(url) as response, tmp_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
            
            # Atomic swap
            tmp_path.replace(dest_path)
            logger.debug(f"Successfully downloaded {url} to {dest_path}")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt == max_retries:
                break
            
            sleep_time = backoff_factor ** attempt
            logger.warning(
                f"Transient error downloading {url}: {e}. "
                f"Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    logger.error(f"Failed to download {url} after {max_retries} attempts.")
    if last_error:
        raise last_error
    raise RuntimeError(f"Unknown error downloading {url}")
