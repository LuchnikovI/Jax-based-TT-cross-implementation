import jax.numpy as jnp
from jax import random
from utils import _random_indices_subset, _concat_indices, _left_skeleton, _right_skeleton, truncate


class TT:

    def __init__(self,
                 key,
                 max_r,
                 shape):
        """Tensor-train for TT-cross interpolation.

        key: PRNGKey
        max_r: int value, maximal TT rank
        shape: tuple, shape of a tensor"""

        self.current_sweep = 0  # number of dmrg sweeps
        self.mode = 'fwd'  # current sweep mode
        self.kernel_num = 0  # current kernel number
        self.max_r = max_r
        self.shape = shape
        self.kernels = len(shape) * [None]  # TT kernels
        key, subkey = random.split(key)
        keys = random.split(subkey, len(shape)-1)
        dim = shape[0]
        self.left_indices = [jnp.arange(dim)[:, jnp.newaxis]]  # left set of indices
        for i, dim in enumerate(shape[1:-1]):
            new_indices = _concat_indices(self.left_indices[-1], jnp.arange(dim)[:, jnp.newaxis])
            if new_indices.shape[0] >= max_r:
                new_indices = _random_indices_subset(keys[i], new_indices, max_r)
            self.left_indices.append(new_indices)
        key, subkey = random.split(key)
        keys = random.split(subkey, len(shape)-1)
        shape = shape[::-1]
        dim = shape[0]
        self.right_indices = [jnp.arange(dim)[:, jnp.newaxis]]  # right set of indices
        for i, dim in enumerate(shape[1:-1]):
            new_indices = _concat_indices(jnp.arange(dim)[:, jnp.newaxis], self.right_indices[-1])
            if new_indices.shape[0] >= max_r:
                new_indices = _random_indices_subset(keys[i], new_indices, max_r)
            self.right_indices.append(new_indices)
        self.right_indices = self.right_indices[::-1]

    def get_arg(self):
        """The method returns the current set of arguments (array of shape
        (batch_size, number_of_modes)) that needs to be 'measured'."""

        if self.kernel_num == 0:
            central_index = jnp.arange(self.shape[0])[:, jnp.newaxis]
            return _concat_indices(central_index, self.right_indices[0])
        elif self.kernel_num == len(self.shape) - 1:
            central_index = jnp.arange(self.shape[-1])[:, jnp.newaxis]
            return _concat_indices(self.left_indices[-1], central_index)
        else:
            central_index = jnp.arange(self.shape[self.kernel_num])[:, jnp.newaxis]
            return _concat_indices(_concat_indices(self.left_indices[self.kernel_num-1], central_index), self.right_indices[self.kernel_num])

    def update(self,
               measurements,
               eps=1e-3):
        """The method updates the current kernel based on the obtained measurements.

        Args:
            measurements: array of shape (batch_size,)
            eps: float value, accuracy"""

        if self.mode == 'fwd':
            if self.kernel_num == 0:
                measurements = measurements.reshape((1, self.shape[0], -1))
                new_kernel, indices = _left_skeleton(measurements, eps)
                new_kernel = new_kernel.reshape((1, self.shape[0], -1))
                self.kernels[0] = new_kernel
                # self.left_indices[0] = jnp.arange(self.shape[0])[:, jnp.newaxis][indices]
                self.kernel_num += 1
            elif self.kernel_num == len(self.shape) - 1:
                self.kernels[-1] = measurements.reshape((-1, self.shape[-1], 1))
                self.mode = 'rev'
                self.current_sweep +=1
            else:
                r_left = self.left_indices[self.kernel_num-1].shape[0]
                r_right = self.right_indices[self.kernel_num].shape[0]
                dim = self.shape[self.kernel_num]
                measurements = measurements.reshape((r_left, dim, -1))
                new_kernel, indices = _left_skeleton(measurements, eps)
                new_kernel = new_kernel.reshape((r_left, dim, -1))
                self.kernels[self.kernel_num] = new_kernel
                self.left_indices[self.kernel_num] = _concat_indices(self.left_indices[self.kernel_num-1], jnp.arange(dim)[:, jnp.newaxis])[indices]
                self.kernel_num += 1
        elif self.mode == 'rev':
            if self.kernel_num == len(self.shape) - 1:
                measurements = measurements.reshape((-1, self.shape[-1], 1))
                new_kernel, indices = _right_skeleton(measurements, eps)
                new_kernel = new_kernel.reshape((-1, self.shape[-1], 1))
                self.kernels[-1] = new_kernel
                # self.right_indices[-1] = jnp.arange(self.shape[-1])[:, jnp.newaxis][indices]
                self.kernel_num -= 1
            elif self.kernel_num == 0:
                self.kernels[0] = measurements.reshape((1, self.shape[0], -1))
                self.mode = 'fwd'
                self.current_sweep +=1
            else:
                r_left = self.left_indices[self.kernel_num-1].shape[0]
                r_right = self.right_indices[self.kernel_num].shape[0]
                dim = self.shape[self.kernel_num]
                measurements = measurements.reshape((-1, dim, r_right))
                new_kernel, indices = _right_skeleton(measurements, eps)
                new_kernel = new_kernel.reshape((-1, dim, r_right))
                self.kernels[self.kernel_num] = new_kernel
                self.right_indices[self.kernel_num-1] = _concat_indices(jnp.arange(dim)[:, jnp.newaxis], self.right_indices[self.kernel_num])[indices]
                self.kernel_num -= 1

    def eval(self,
             indices):
        """The method evaluates TT at a given set of indices (arguments).

        Args:
            indices: array of shape (batch_size, number_of_modes), arguments

        Returns:
            array of shape (batch_size,), results"""
        left = self.kernels[0][:, indices[:, 0]].transpose((1, 0, 2))
        log_norm = 0.
        for i, kernerl in enumerate(self.kernels[1:]):
            left = left @ (kernerl[:, indices[:, i+1]].transpose((1, 0, 2)))
            norm = jnp.linalg.norm(left)
            left /= norm
            log_norm += jnp.log(norm)
        return left.reshape((-1,)) * jnp.exp(log_norm)

    def compression(self, 
                    eps):
        """The method compress the TT decomposition of a tensor.

        Args:
            eps: accuracy of a local truncation
    
        Returns: infidelity"""

        self.kernels, infidelity = truncate(self.kernels, eps)
        return infidelity
