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
        import icl_dl_cw2_utils  # noqa: F401
        return
    except Exception:
        pass
    os.makedirs(_CACHE, exist_ok=True)
    data = urllib.request.urlopen(_URL, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(_CACHE)
    for n in os.listdir(_CACHE):
        if n.startswith("icl_dl_cw2_utils") and n != "icl_dl_cw2_utils":
            link = os.path.join(_CACHE, "icl_dl_cw2_utils")
            if not os.path.exists(link):
                os.symlink(os.path.join(_CACHE, n), link)
    sys.path.insert(0, _CACHE)


def get_dataset(data_root, transform, split="train"):
    ensure_utils()
    from icl_dl_cw2_utils.utils.hotdogdataset import DLHotDogDataset
    return DLHotDogDataset(data_root, transform=transform, split=split)


def get_classifier(device, checkpoint_path="./_deps/hotdog_ckpt.pth"):
    ensure_utils()
    from icl_dl_cw2_utils.utils.food_resnet import HotDogClassifier
    return HotDogClassifier(checkpoint_path=checkpoint_path, device=device)
