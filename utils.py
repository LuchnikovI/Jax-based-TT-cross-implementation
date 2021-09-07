import jax.numpy as jnp
from jax import jit, random, lax
from jax.ops import index, index_update


def _concat_indices(indices1, indices2):
    length1 = indices1.shape[1]
    length2 = indices2.shape[1]
    indices2 = jnp.tile(indices2.reshape((1, -1, length2)), (indices1.shape[0], 1, 1))
    indices1 = jnp.tile(indices1.reshape((-1, 1, length1)), (1, indices2.shape[1], 1))
    indices = jnp.concatenate([indices1, indices2], axis=-1)
    indices = indices.reshape((-1, indices.shape[2]))
    return indices

def _random_indices_subset(key, indices, r):
    mask = jnp.concatenate([jnp.ones((r,), bool),
                            jnp.zeros((indices.shape[0] - r), bool)], axis=0)
    mask = random.permutation(key, mask)
    _, key = random.split(key)
    return random.permutation(key, indices[mask])

@jit
def _maxvol(A, eps):
    def iter(vars):
        A, b_max, order = vars
        a = A[:A.shape[1]]
        B = A @ jnp.linalg.inv(a)
        ind = jnp.argmax(jnp.abs(B))
        inds = jnp.unravel_index(ind, A.shape)
        b_max = jnp.abs(B[inds[0], inds[1]])
        aux_slice = A[inds[0], :]
        aux_index = order[inds[0]]
        order = index_update(order, index[inds[0]], order[inds[1]])
        order = index_update(order, index[inds[1]], aux_index)
        A = index_update(A, index[inds[0], :], A[inds[1], :])
        A = index_update(A, index[inds[1], :], aux_slice)
        return A, b_max, order
    def cond(vars):
        _, b_max, _ = vars
        return b_max > 1 + eps
    _, _, order = lax.while_loop(cond, iter, (A, 1e16, jnp.arange(A.shape[0])))
    order = jnp.arange(A.shape[0])[order]
    order = order[:A.shape[1]]
    return order


@jit
def _left_skeleton(unfolding, eps):
    r, dim, _ = unfolding.shape
    unfolding = unfolding.reshape((r * dim, -1))
    q, _ = jnp.linalg.qr(unfolding)
    indices = maxvol(q, eps)
    q_hat = q[indices]
    new_kernel = q @ jnp.linalg.inv(q_hat)
    new_kernel = new_kernel.reshape((r, dim, -1))
    return new_kernel, indices


@jit
def _right_skeleton(unfolding, eps):
    _, dim, r = unfolding.shape
    unfolding = unfolding.reshape((-1, r * dim))
    unfolding = unfolding.T
    q, _ = jnp.linalg.qr(unfolding)
    indices = maxvol(q, eps)
    q_hat = q[indices]
    new_kernel = q @ jnp.linalg.inv(q_hat)
    new_kernel = new_kernel.T
    new_kernel = new_kernel.reshape((-1, dim, r))
    return new_kernel, indices


def maxvol(A, eps):
    """Implementation of the maxvol algorithm.

    Args:
        A: array of shape (n, r), matrix for which one needs to
            find a maxvol submatrix
        eps: float value, accuracy

    Returns:
        array of shape (r,), indices that form
        a maxvol matrix, i.e. A[indices] is a maxvol
        matrix"""

    if A.shape[0] < A.shape[1]:
        return jnp.arange(A.shape[0])
    else:
        return _maxvol(A, eps)
