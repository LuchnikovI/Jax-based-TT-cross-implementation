import jax.numpy as jnp
from jax import jit, random, lax
from jax.ops import index, index_update
from functools import reduce


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
    order = order[:A.shape[1]]
    return order


def _left_skeleton(unfolding, eps):

    r, dim, _ = unfolding.shape
    unfolding = unfolding.reshape((r * dim, -1))
    q, _ = jnp.linalg.qr(unfolding)
    indices = maxvol(q, eps)
    q_hat = q[indices]
    new_kernel = q @ jnp.linalg.inv(q_hat)
    new_kernel = new_kernel.reshape((r, dim, -1))
    return new_kernel, indices


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
        A: complex or real valued array of shape (n, m)
        eps: real valued number representing accuracy of the algorithm.

    Returns:
        int valued array of shape (m,) representing numbers of rows
        that are forming the maxvol submatrix"""

    if A.shape[0] < A.shape[1]:
        return jnp.arange(A.shape[0])
    else:
        return _maxvol(A, eps)


def _set_left_canonical(kernels):

    def push_r_right(vars, ker):
        updated_state, log_norm, r = vars
        ker = jnp.tensordot(r, ker, axes=1)
        left_bond, dim, _ = ker.shape
        ker = ker.reshape((left_bond*dim, -1))
        ker, r = jnp.linalg.qr(ker)
        ker = ker.reshape((left_bond, dim, -1))
        norm = jnp.linalg.norm(r)
        r /= norm
        updated_state += [ker]
        log_norm += jnp.log(norm)
        return updated_state, log_norm, r
    return reduce(push_r_right, [([], jnp.array(0.), jnp.array([[1.]]))] + kernels)


def _truncate_left_canonical(kernels,
                             r,
                             log_norm,
                             eps):

    scale = jnp.exp(log_norm / len(kernels))  # rescaling coeff. for tt kernels
    def push_r_left(vars, ker):
        updated_state, r = vars
        left_bond, dim, _ = ker.shape
        ker = jnp.tensordot(ker, r, axes=1)
        ker = ker.reshape((left_bond, -1))
        u, s, ker = jnp.linalg.svd(ker)
        # setting threshold
        sq_norm = (s ** 2).sum()
        cum_sq_norm = jnp.cumsum((s ** 2)[::-1])
        trshld = (jnp.sqrt(cum_sq_norm / sq_norm) > eps).sum()
        # truncation
        u = u[:, :trshld]
        s = s[:trshld]
        ker = ker[:trshld]
        ker = ker.reshape((trshld, dim, -1))
        r = u * s
        updated_state = [scale * ker] + updated_state
        return updated_state, r
    return reduce(push_r_left, [([], r)] + kernels[::-1])


def truncate(kernels, eps):
    """Truncates TT decomposition of a tensor.

    Args:
        kernels: list with TT kernels.
        eps: real valued number representing accuracy of the local truncation.

    Returns:
        kernels: list with truncated TT kernels.
        infidelity: real valued number representing final infidelity"""

    kernels, log_norm, r = _set_left_canonical(kernels)
    kernels, norm = _truncate_left_canonical(kernels, r, log_norm, eps)
    kernels[0] *= norm / jnp.abs(norm)
    infidelity = jnp.sqrt(jnp.abs(1 - norm[0, 0] ** 2))
    return kernels, infidelity
