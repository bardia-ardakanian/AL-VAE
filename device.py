import os
from distutils.version import LooseVersion

import pkg_resources


def assert_torch_installed():
    # Check that torch version 1.6.0 or later has been installed. If not, refer
    # user to the PyTorch webpage for install instructions.
    torch_pkg = None
    try:
        torch_pkg = pkg_resources.get_distribution("torch")
    except pkg_resources.DistributionNotFound:
        pass
    assert torch_pkg is not None, (  # and LooseVersion(torch_pkg.version) >= LooseVersion("1.6.0")
        "A compatible version of PyTorch was not installed. Please visit the PyTorch homepage "
        + "(https://pytorch.org/get-started/locally/) and follow the instructions to install. "
        + "Version 1.6.0 and later are supported."
    )


assert_torch_installed()

# This should be the only place that we import torch directly.
# Everywhere else is caught by the banned-modules setting for flake8
import torch  # noqa I201
from torch.overrides import TorchFunctionMode

_DEVICE_CONSTRUCTOR = {
    # standard ones
    torch.empty,
    torch.empty_strided,
    torch.empty_quantized,
    torch.ones,
    torch.arange,
    torch.bartlett_window,
    torch.blackman_window,
    torch.eye,
    torch.fft.fftfreq,
    torch.fft.rfftfreq,
    torch.full,
    torch.fill,
    torch.hamming_window,
    torch.hann_window,
    torch.kaiser_window,
    torch.linspace,
    torch.logspace,
    torch.nested.nested_tensor,
    # torch.normal,
    torch.ones,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.range,
    torch.sparse_coo_tensor,
    torch.sparse_compressed_tensor,
    torch.sparse_csr_tensor,
    torch.sparse_csc_tensor,
    torch.sparse_bsr_tensor,
    torch.sparse_bsc_tensor,
    torch.tril_indices,
    torch.triu_indices,
    torch.vander,
    torch.zeros,
    torch.asarray,
    # weird ones
    torch.tensor,
    torch.as_tensor,
}

torch.set_num_threads(2)
os.environ["KMP_BLOCKTIME"] = "0"


class DeviceMode(TorchFunctionMode):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _DEVICE_CONSTRUCTOR and 'device' not in kwargs:
            kwargs['device'] = self.device
        return func(*args, **kwargs)


_device = torch.device("cpu")
_device_mode = None


def set_torch_config():
    global _device

    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    _device = torch.device(device_str)

    if _device.type == "mps":
        global _device_mode
        _device_mode = DeviceMode(_device)
        _device_mode.__enter__()

    if _device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)


# Initialize to default settings
set_torch_config()

nn = torch.nn

def default_device():
    set_torch_config()
    print(_device)

    return _device
