import numpy as np
from boat_submission import iter_ancestors


def build_cov_matrix(tree, leaves, bursts, var_p, var_d):
    """Return a covariance matrix given the phylogeny and burst locations.

    tree - tree processed by number_tree(...)
    leaves - list created by number_leaves(...)
    bursts - 1D np array denoting the number of bursts along the i^th edge.
    var_p - population variance
    var_d - displacement variance
    """

    # Total number of leaves
    n_leaves = len(leaves)

    # Covariance matrix--depends on the number of bursts, etc.
    cov_matrix = np.identity(n_leaves)

    # Diagonal terms--population fluctuations
    cov_matrix = var_p * cov_matrix

    # Diagonal terms--burst fluctuations
    for i in range(n_leaves):
        n_bursts = 0.0

        # Total numer of bursts
        for node in iter_ancestors(leaves[i]):
            n_bursts += bursts[node.id]

        # Var(X_i) = var_p + var_d * (# of bursts in ancestry)
        cov_matrix[i][i] += var_d * n_bursts

    # Off-diagonal terms--these correspond only to burst fluctuations
    for i in range(1, n_leaves):
        for j in range(0, i):
            common_ancestor = tree.get_common_ancestor(leaves[i], leaves[j])

            n_bursts = 0.0

            for node in iter_ancestors(common_ancestor):
                n_bursts += bursts[node.id]

            cov_matrix[i][j] = var_d * n_bursts
            # Covariance marix is symmetric
            cov_matrix[j][i] = cov_matrix[i][j]

    return cov_matrix


def build_bm_cov_matrix(tree, leaves, var_p, var_bm):
    """Return a covariance matrix given the phylogeny, for a BM + white noise
    model. Used for comparison.

    tree - tree processed by number_tree(...).
    leaves - list created by number_leaves(...).
    var_p - population variance.
    var_bm - variance of BM component.
    """

    # Total number of leaves
    n_leaves = len(leaves)

    # Covariance matrix--depends on the number of bursts, etc.
    cov_matrix = np.identity(n_leaves)

    # Diagonal terms--population fluctuations
    cov_matrix = var_p * cov_matrix

    # Diagonal terms--burst fluctuations
    for i in range(n_leaves):
        lineage_length = 0.0

        # Total numer of bursts
        for node in iter_ancestors(leaves[i]):
            lineage_length += node.dist

        # Var(X_i) = var_p + var_d * (# of bursts in ancestry)
        cov_matrix[i][i] += var_bm * lineage_length

    # Off-diagonal terms--these correspond only to burst fluctuations
    for i in range(1, n_leaves):
        for j in range(0, i):
            common_ancestor = tree.get_common_ancestor(leaves[i], leaves[j])

            common_lineage_length = 0.0

            for node in iter_ancestors(common_ancestor):
                common_lineage_length += node.dist

            cov_matrix[i][j] = var_bm * common_lineage_length
            # Covariance marix is symmetric
            cov_matrix[j][i] = cov_matrix[i][j]

    return cov_matrix
