import jax.numpy as jnp
from jax import random
from utils import _random_indices_subset, _concat_indices, _left_skeleton, _right_skeleton, truncate


class TT_cross:

    def __init__(self,
                 key,
                 max_r,
                 shape):
        """Tensor-train.

        key: PRNGKey
        max_r: int valued number representing maximal TT rank.
        shape: tuple representing shape of a tensor."""

        self.current_sweep = 0  # number of a current dmrg sweeps
        self.mode = 'fwd'  # current sweep mode (fwd or rev)
        self.kernel_num = 0  # current kernel number
        self.max_r = max_r  # maximal TT rank
        self.shape = shape  # shape of a tensor
        self.kernels = len(shape) * [None]  # list of TT kernels

        # initial random indices representing maxvol of an unfolding matrix
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
        """Provides a current set of arguments which have to be evaluated (measured).

        Returns:
            int valued array of shape (batch_size, number_of_modes)."""

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
        """Updates the current kernel based on the obtained measurements.

        Args:
            measurements: array of shape (batch_size,).
            eps: float valued number representing accuracy."""

        if self.mode == 'fwd':
            if self.kernel_num == 0:
                measurements = measurements.reshape((1, self.shape[0], -1))
                new_kernel, indices = _left_skeleton(measurements, eps)
                new_kernel = new_kernel.reshape((1, self.shape[0], -1))
                self.kernels[0] = new_kernel
                self.left_indices[0] = jnp.arange(self.shape[0])[:, jnp.newaxis][indices]
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
                self.right_indices[-1] = jnp.arange(self.shape[-1])[:, jnp.newaxis][indices]
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

    def get_tt_kernels(self):
        """Returns list with TT kernels."""

        return self.kernels

    def random_args(self, key, n):
        """Samples set of random arguments.

        Args:
            key: PRNGKey.
            n: int valued number representing number of arguments.

        Returns:
            array of shape (n, number_of_modes) representing set of arguments."""

        return jnp.concatenate([jnp.argmax(random.gumbel(key, (n, 1, dim)), axis=2) for dim in self.shape], axis=1)


# Functions that are necessary for operating with tensors in TT format


def eval(tt_kernels,
         indices):
    """Evaluates TT at a given set of indices (arguments).

    Args:
        tt_kernels: list with TT kernels.
        indices: array of shape (batch_size, number_of_modes) representing arguments.

    Returns:
        array of shape (batch_size,) representing the results of ecaluation."""

    left = tt_kernels[0][:, indices[:, 0]].transpose((1, 0, 2))
    log_norm = 0.
    for i, kernerl in enumerate(tt_kernels[1:]):
        left = left @ (kernerl[:, indices[:, i+1]].transpose((1, 0, 2)))
        norm = jnp.linalg.norm(left)
        left /= norm
        log_norm += jnp.log(norm)
    return left.reshape((-1,)) * jnp.exp(log_norm)


def compression(tt_kernels,
                eps):
    """Compresses the TT representation of a tensor.

    Args:
        tt_kernels: list with TT kernels.
        eps: real valued number representing truncation accuracy."""

    return truncate(tt_kernels, eps)


def dot(tt_kernels_1,
        tt_kernels_2):
    """Evaluates the dot product between two tensors in TT format.

    Args:
        tt_kernels_1: list with TT kernels representing the first tensor.
        tt_kernels_2: list with TT kernels representing the second tensor.

    Returns:
        the result of the dot product in the following terms:
            log_abs: log(|z|),
            phi: phase of z."""

    tt_kernels_1 = list(map(jnp.conj, tt_kernels_1))
    left = jnp.tensordot(tt_kernels_1[0], tt_kernels_2[0], axes=[[1], [1]])
    left = left[0, :, 0]
    log_norm = 0.
    for i, (kernel_1, kernel_2) in enumerate(zip(tt_kernels_1[1:], tt_kernels_2[1:])):
        left = jnp.tensordot(left, kernel_1, axes=[[0], [0]])
        norm = jnp.linalg.norm(left)
        left /= norm
        log_norm += jnp.log(norm)
        left = jnp.tensordot(left, kernel_2, axes=[[0, 1], [0, 1]])
        norm = jnp.linalg.norm(left)
        left /= norm
        log_norm += jnp.log(norm)
    return log_norm, jnp.angle(left)[0, 0]


def relative_difference_sq(tt_kernels_1,
                           tt_kernels_2):
    """Calculates square of the relative difference between tensors.

    Args:
        tt_kernels_1: list with TT kernels representing the first tensor.
        tt_kernels_2: list with TT kernels representing the second tensor.

    Returns:
        real valued number representing square of the relative difference."""

    log_abs_11, _ = dot(tt_kernels_1, tt_kernels_1)
    log_abs_22, _ = dot(tt_kernels_2, tt_kernels_2)
    log_abs_12, _ = dot(tt_kernels_1, tt_kernels_2)
    diff_sq = 1 + jnp.exp(log_abs_22 - log_abs_11) - 2 * jnp.exp(log_abs_12 - log_abs_11)
    return diff_sq
