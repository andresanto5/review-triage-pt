"""Download the public dataset and verify the experiment fingerprint."""
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv'
EXPECTED = '821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38'


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    target = ROOT/'data/raw/B2W-Reviews01.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if fingerprint(target) != EXPECTED:
            raise RuntimeError('Existing dataset differs; not overwritten.')
        print('Dataset already present; SHA-256 verified.')
        return
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            with urlopen(URL, timeout=120) as response:
                shutil.copyfileobj(response, temp)
        if fingerprint(temp_path) != EXPECTED:
            raise RuntimeError('Upstream dataset hash changed; inspect before repeating the experiment.')
        temp_path.replace(target)
        print('Dataset downloaded; SHA-256 verified. License: CC BY-NC-SA 4.0, B2W Digital.')
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
