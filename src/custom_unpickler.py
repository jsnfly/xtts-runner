import pickle
import torch
import numpy as np


class CustomDict(dict):
    @property
    def __dict__(self):
        return self


class TensorWrapper:
    def __init__(self, size, dtype=torch.float32, device='cpu'):
        self.size = size
        self.dtype = dtype
        self.device = device

    def set_(self, storage, offset, size, stride):
        if not isinstance(size, tuple):
            size = tuple(size)
        # Handle empty size case
        if len(size) == 0:
            return torch.tensor(storage.data[offset], dtype=storage.dtype)
        # Handle scalar case
        if size == (1,):
            return torch.tensor(storage.data[offset], dtype=storage.dtype)

        # Calculate total number of elements needed based on size
        total_elements = np.prod(size)

        # Slice the storage data using the offset
        data_slice = storage.data[offset:offset + total_elements]

        # Regular case with proper offset handling
        return torch.from_numpy(data_slice).to(storage.dtype).view(size)


class CustomStorage:
    def __init__(self, data, dtype):
        self.dtype = dtype
        self.data = data

    @property
    def _untyped_storage(self):
        return self

    @property
    def device(self):
        return 'cpu'


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that substitutes dummy classes for missing ones."""

    DTYPE_MAP = {
        'FloatStorage': (torch.float32, np.float32),
        'LongStorage': (torch.int64, np.int64),
        'IntStorage': (torch.int32, np.int32),
        'BoolStorage': (torch.bool, np.bool_),
    }

    def __init__(self, file, zip_archive):
        super().__init__(file)
        self.zip_archive = zip_archive

    def persistent_load(self, pid):
        """Handle persistent ID loading by returning a dummy storage with actual data."""
        if isinstance(pid, tuple) and pid[0] == 'storage':
            storage_type, storage_class, key, location, numel = pid

            # Handle the case where storage_class is already CustomStorage
            if isinstance(storage_class, type) and storage_class.__name__ == 'CustomStorage':
                storage_name = 'FloatStorage'  # default to float if we get CustomStorage
            else:
                storage_name = storage_class.__name__

            torch_dtype, np_dtype = self.DTYPE_MAP.get(storage_name, (torch.float32, np.float32))

            try:
                tensor_data = self.load_tensor_data(key, numel, np_dtype)
                return CustomStorage(tensor_data, dtype=torch_dtype)
            except Exception as e:
                print(f"Failed to load tensor data: {e}")
                return CustomStorage(
                    np.zeros(numel, dtype=np_dtype),
                    dtype=torch_dtype
                )

        return pid[1]

    def load_tensor_data(self, key, numel, dtype):
        """Load tensor data from the zip file."""
        data_file = f'model/data/{key}'
        with self.zip_archive.open(data_file, 'r') as f:
            data_bytes = f.read()
            return np.frombuffer(data_bytes, dtype=dtype)

    def find_class(self, module, name):
        """Override find_class to return dummy classes for missing ones."""
        try:
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return self._rebuild_tensor_v2
            if module == 'torch._utils' and name == '_rebuild_tensor':
                return self._rebuild_tensor
            if module == 'torch' and name in self.DTYPE_MAP:
                return CustomStorage
            return super().find_class(module, name)
        except:
            return CustomDict

    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        """Custom tensor rebuilding function."""
        tensor = self._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        return tensor

    def _rebuild_tensor(self, storage, storage_offset, size, stride):
        """Create a new tensor with the given size and data."""
        wrapper = TensorWrapper(size, dtype=storage.dtype)
        return wrapper.set_(storage, storage_offset, size, stride)
