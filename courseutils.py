"""Make the course utilities (`icl_dl_cw2_utils`) importable.

The utils repo is a namespace package with no setup.py, so it can't be pip-installed
normally. This helper imports it if already on the path, otherwise downloads a snapshot
and adds it to sys.path. Provides the dataset and the public classifier.
"""
import io
import os
import sys
import tarfile
import urllib.request

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deps")
_URL = "https://codeload.github.com/bkainz/icl_dl_cw2_utils/tar.gz/HEAD"


def ensure_utils():
    try:
        from utils import hotdogdataset  # noqa: F401
        return
    except Exception:
        pass
    os.makedirs(_CACHE, exist_ok=True)
    data = urllib.request.urlopen(_URL, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(_CACHE)
    for name in os.listdir(_CACHE):
        root = os.path.join(_CACHE, name)
        if name.startswith("icl_dl_cw2_utils") and os.path.isdir(root):
            # The upstream archive contains a top-level ``utils`` package.
            # Adding its root to sys.path works on Linux and Windows; unlike
            # the starter's symlink approach, it also works from a WSL UNC
            # path without Windows Developer Mode/administrator privileges.
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("could not locate the downloaded coursework utilities")


def get_dataset(data_root, transform, split="train"):
    ensure_utils()
    from utils.hotdogdataset import DLHotDogDataset
    return DLHotDogDataset(data_root, transform=transform, split=split)


def get_classifier(device, checkpoint_path="./_deps/hotdog_ckpt.pth"):
    ensure_utils()
    from utils.food_resnet import HotDogClassifier
    return HotDogClassifier(checkpoint_path=checkpoint_path, device=device)
