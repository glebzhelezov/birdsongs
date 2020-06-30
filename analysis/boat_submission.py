""" BOAT: Birds On A Tree module

This module allows you to compute the likelihood of phenotype means on the leaves of an
evolutionary tree, given a burst-on-edges model. A better name would be "Bursts On A Tree."

Please direct questions and comments to:
1. Jay McEntee (U of Florida) at jaymcentee@ufl.edu (esp. re: evo-bio, analysis)
2. Gleb Zhelezov (U of Edinburgh) at gleb.zhelezov@ed.ac.uk (esp. re: math model, code)

            ((
            \\``.
            \_`.``-. 
            ( `.`.` `._  
             `._`-.    `._ 
               \`--.   ,' `. 
                `--._  `.  .`. 
                 `--.--- `. ` `. 
                 `.--  `;  .`._ 
                   :-   :   ;. `.__,.,__ __    First one in the family!
                    `\  :       ,-(     ';o`>.  /
                      `-.`:   ,'   `._ .:  (,-`,
                     \    ;      ;.  ,: 
                     ,"`-._>-:        ;,'  `---.,---.
                     `>'"  "-`       ,'   "":::::".. `-.
                      `;"'_,  (\`\ _ `:::::::::::'"     `---.
                  -hrr-    `-(_,' -'),)\`.       _      .::::"'  `----._,-"")
                       \_,': `.-' `-----' `--;-.   `.   ``.`--.____/ 
                         `-^--'                \(-.  `.``-.`-=:-.__)
                                            `  `.`.`._`.-._`--.)
                                                 `-^---^--.`--
            .
            |\
            | \
            |  \
            |   \
            |    \
            |     \ What a wingspan!
            |      \       /
            |       \     / ____O
            |        \     .' ./
            |   _.,-~"\  .',/~'
            <-~"   _.,-~" ~ |
^"~-,._.,-~"^"~-,._\       /,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._
~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._
^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._
~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._

(ASCII art found online.)

"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from scipy.linalg import pinv
from scipy.linalg import det
from scipy.linalg import LinAlgError
from scipy.optimize import minimize, brute, basinhopping, minimize_scalar
from ete3 import Tree, TextFace
from itertools import combinations_with_replacement
from collections import Counter
from collections import OrderedDict
from joblib import Parallel, delayed
from csv import reader
from enum import Enum, auto
from datetime import datetime
from numbers import Number

# We'll use this to denote under which mode to calculate likelihood, etc.
class Models(Enum):
    PULSE = auto()
    PULSE_BM = auto()
    BM = auto()
    PULSE_SEPVARP = auto()
    PULSE_MU = auto()


def number_leaves(in_tree):
    """Number the leaves, count their total number, and create a list of pointers to each leaf.

    Returns n_leaves, leaves, where:
    n_leave - total number of leaves
    leaves - list of pointers to each leaf
    """
    leaves = []
    n_leaves = 0
    for leaf in in_tree.iter_leaves():
        leaves.append(leaf)
        leaf.add_feature("leaf_id", n_leaves)
        n_leaves = n_leaves + 1

    return n_leaves, leaves


def number_edges(in_tree):
    """Get total number of edges, number them, and create a list of pointers to each edge.
    
    ** Update 28/03/2019: We now add an id to the top node, but don't count it!
    ** This is to simplify calculations by increments!
    We associate with each node the edge above it, and exclude the root node.
    node[] gives the node, given the id; node.id gives the id, given the node.
    We compute the sum of all the distances, to make burst probability
    calculations faster.

    Returns n_nodes, nodes, distances, where:
    n_nodes - total number of node
    nodes - list of pointers to each node
    distances - time length associated with i^th node
    """
    n_nodes = 0
    nodes = []
    distances = []

    # for node in t.traverse("postorder"):
    for node in in_tree.iter_descendants("postorder"):
        nodes.append(node)
        node.add_feature("id", n_nodes)
        distances.append(node.dist)
        n_nodes = n_nodes + 1

    # Give top node an ID, too
    in_tree.add_feature("id", n_nodes)

    # Sum of all distances
    distances_sum = sum(distances)
    in_tree.add_feature("distances_sum", distances_sum)

    return n_nodes, nodes, distances


def iter_ancestors(node):
    """Generator for all ancestors of a given node, excluding root node.

    This is used to find the common ancestry, in order to compute the covariance between leaves.
    """
    while node.up != None:
        yield node
        node = node.up


def burst_configs(n_edges, max_bursts, min_bursts=0):
    """Generator which returns locations of bursts.

    Returns a list of this sort: [(edge #, # of bursts on edge), ..., (edge #, # bursts)]. List only
    includes tuples with nonzero numbers of bursts.
    """

    for n_bursts in range(min_bursts, max_bursts + 1):
        # Can run this in parallel
        for individual_bursts in combinations_with_replacement(
            range(n_edges), n_bursts
        ):
            burst_locations = []

            # This is probably the worst way to generate a list of the form
            # [(edge_index_1, n_bursts_1), ..., (edge_index_k, n_bursts_k)]
            for burst in Counter(individual_bursts).items():
                burst_locations.append((burst[0], burst[1]))

            # Yield a list of all the burst locations in this configuration.
            yield burst_locations


def decorate_tree(input_tree):
    """Labels all the nodes and edges on the graph of a processed tree. Call input_tree.show() to see."""
    for node in input_tree.iter_descendants():
        node.add_face(TextFace(node.id), column=0, position="branch-top")

    for leaf in input_tree.iter_leaves():
        leaf.add_face(
            TextFace(leaf.leaf_id, fgcolor="red", fsize=6),
            column=0,
            position="branch-bottom",
        )


# Helper functions for setting initial guesses for optimization
def n_bursts_estimate(data):
    # Find some "natural breaks" to count the number of bursts (v. heuristic)
    diffsqrs = []

    for i in range(len(data) - 1):
        diffsqrs.append((data[i + 1] - data[i]) ** 2)

    threshold = 2.5 * np.average(diffsqrs)

    n_gaps = 0

    for diffsqr in diffsqrs:
        if diffsqr > threshold:
            n_gaps = n_gaps + 1

    return n_gaps


def sigma_p_estimate(data):
    running_sum = 0.0

    for i in range(int(len(data) / 2)):
        a = data[2 * i]
        b = data[2 * i + 1]
        running_sum += 0.5 * (a - b) ** 2

    return 2.0 * running_sum / len(data)


def sigma_d_estimate(data):
    diffsqr = 0.5 * (max(data) - min(data)) ** 2
    # Rough estimate of number of bursts
    n_gaps = n_bursts_estimate(data)

    if n_gaps < 1:
        return diffsqr / 10.0

    return diffsqr / n_gaps


def mean_ancestor_estimate(data):
    return np.average(data)




def regime_mu_hat(var_p, regime, data, errors):
    """Compute MLE for mu within a reigme."""
    mu_hat = 0
    normalizing_constant = 0

    for leaf_id in regime:
        mu_hat += data[leaf_id] / (var_p + errors[leaf_id])
        normalizing_constant += 1. / (var_p + errors[leaf_id])
    
    if normalizing_constant < 1e-20:
        print(normalizing_constant, len(regime))
    mu_hat = mu_hat/normalizing_constant
    
    return mu_hat


def neglkl_regime_var_p(var_p, regime, data, errors):
    """negloglikelihood of var_p within a regime (using MLE for mu)."""
    mu_hat = regime_mu_hat(var_p, regime, data, errors)
    neglkl = 0
    
    for leaf_id in regime:
        neglkl += 0.5*np.log(2*np.pi*(var_p + errors[leaf_id])) + (data[leaf_id] - mu_hat)**2 / (2*(var_p + errors[leaf_id]))
    
    return neglkl


def neglkl_marginal_var_p(var_p, pulse_positions, tree, data, errors):
    """Return negloglikelihood of var_p marginal--i.e. get contribs from all regimes."""
    neglkl = 0
    grouped_leaf_ids = group_ids(group_by_regime(pulse_positions, tree))
    
    for regime in grouped_leaf_ids:
        neglkl += neglkl_regime_var_p(var_p, regime, data, errors)
    
    return neglkl


def _neglkl_marginal_var_p_function(pulse_positions, tree, data, errors):
    """Return marginal var_p netloglikelihood _function_ to minimize."""
    grouped_leaf_ids = group_ids(group_by_regime(pulse_positions, tree))
    
    def _f(var_p):
        neglkl = 0
        for regime in grouped_leaf_ids:
            #print(var_p, regime, data, errors)
            neglkl += neglkl_regime_var_p(var_p, regime, data, errors)
        
        return neglkl
    
    return _f



def KL(S1, mu1, S2, mu2):
    """Compute D_KL(P_1 || P_2)) where both are multivariate normals."""
    S1 = np.array(S1)
    S2 = np.array(S2)
    mu1 = np.array(mu1)
    mu2 = np.array(mu2)
    
    S2inv = np.linalg.pinv(S2)
    
    to_return = 0
    eps = 1e-60
    #to_return += np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1]
    to_return += np.log(np.linalg.det(S2)) - np.log(np.linalg.det(S1))
    to_return -= len(S1.diagonal())
    to_return += (S2inv @ S1).diagonal().sum()
    to_return += (mu2-mu1)@np.linalg.pinv(S2)@(mu2-mu1)
    
    return to_return/2


def optimized_phylo_distances(data, tree, leaves, distances, error_variances=0, n_iterations = 30):
    """Find ''virtual distances'' so that phenotypes evolve correctly, under BM model with var_BM=1.
    Also return the mean.
    
    We need this for computing the optimal covariance matrix. """
    best_distances = np.array(distances)

    dummy_distances = np.array(best_distances)

    f =  distances_negloglkl_fun(data, tree, leaves, error_variances=error_variances)

    best_distances = np.array(distances)
    likelihoods = []

    for _ in range(n_iterations):
        for i in range(len(best_distances)):
            dummy_distances = np.array(best_distances)
            
            def _g(x):
                dummy_distances[i] = x
                return f(dummy_distances)

            # We don't change the branch size too much at once.
            res = minimize(lambda x:_g(x[0]), [best_distances[i]], bounds=[(best_distances[i]/3, best_distances[i]*5/3)])

            best_distances[i] = res.x
    
    # 'True' also returns the mean
    return best_distances, f(best_distances, True)[1]



def load_treefile_datafile(treefile, datafile, return_errors=False):
    """Loads up a Newick tree file and a data CSV file."""
    # Read the tree file
    t = Tree(treefile)

    # Process a copy...
    t_copy = t.copy()
    n_leaves, leaves = number_leaves(t_copy)

    # Read the data file
    data_dictionary = {}
    errors_dictionary = {}

    errors_included = 0

    with open(datafile, "r") as csvfile:
        for row in reader(csvfile, delimiter=","):
            if len(row) != 0:
                data_dictionary[row[0]] = row[1]
            if len(row) > 2:
                errors_included += 1
                errors_dictionary[row[0]] = row[2]

    # Check if measurement errors are included
    if errors_included < n_leaves:
        errors_included = False
    else:
        errors_included = True
        errors = []
        max_error = 0.0

    # Match up data with leaf.
    data = []

    for i in range(n_leaves):
        leaf_name = leaves[i].name

        if leaf_name in data_dictionary:
            data.append(float(data_dictionary[leaf_name]))

            # Put in errors & replace non-floats with the max. Note:
            # infinity is treated as a float here, so be careful.
            if errors_included:
                try:
                    new_error = float(errors_dictionary[leaf_name])
                except ValueError:
                    new_error = -1

                if new_error > max_error:
                    max_error = new_error

                errors.append(new_error)
        else:
            print("Warning! No data for {}.".format(leaf_name))

    if return_errors == True and errors_included == False:
        raise ValueError("Errors not included in data file.")

    # Replace anything not a float with the max error.
    if errors_included:
        for i in range(n_leaves):
            if errors[i] == -1:
                errors[i] = max_error

    if len(data) != n_leaves:
        raise ValueError(
            "Unmatched number of data and leaves,"
            " with n_data = {}, n_leaves = {}. "
            "This could be because the names in the two "
            "files are not the same!".format(len(data), n_leaves)
        )

    if return_errors == False:
        return t, data
    else:
        return t, data, errors


def build_phylo_cov_matrix(tree, leaves, phylo_distances):
    """Return a covariance matrix given the phylogeny, for a BM + white noise
    model. Used for comparison.

    tree - tree processed by number_tree(...).
    leaves - list created by number_leaves(...).
    errors - d
    phylo_distances - output of def optimized_phylo_distances(...)
    """

    # Total number of leaves
    n_leaves = len(leaves)

    cov_matrix = np.zeros([n_leaves, n_leaves])

    # Diagonal terms--burst fluctuations
    for i in range(n_leaves):
        lineage_length = 0.0

        # Total numer of bursts
        for node in iter_ancestors(leaves[i]):
            lineage_length += phylo_distances[node.id]

        # Var(X_i) = var_p + var_d * (# of bursts in ancestry)
        cov_matrix[i][i] += lineage_length

    # Off-diagonal terms--these correspond only to burst fluctuations
    for i in range(1, n_leaves):
        for j in range(0, i):
            common_ancestor = tree.get_common_ancestor(leaves[i], leaves[j])

            common_lineage_length = 0.0

            for node in iter_ancestors(common_ancestor):
                common_lineage_length += phylo_distances[node.id]

            cov_matrix[i][j] = common_lineage_length
            # Covariance marix is symmetric
            cov_matrix[j][i] = cov_matrix[i][j]

    return cov_matrix
    
    
def distances_negloglkl_fun(data, tree, leaves, error_variances=0):
    """Return a function which computes the likelihood of the data, under
    BM model with var_bm=1 and variable branch lengths (this is to find a
    good candiate for computing KL divergence.

    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances/const variances.
    """

    # If measurement error variances are constant throughout...
    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))



    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    
    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    # No BM variance at the top--but need to add a virtual 0-long edge.

    def neg_log_lkl_function(distances, return_ml_mean=False):
        distances = np.append(distances, 0)

        # Define variance matrix for this model
        edge_variances = np.array(distances)

        # Add in measurement errors.
        for leaf in leaves:
            # The errors are numbered according to leaf ID. So we get
            # each leaf's ID, get the error, then append it to the
            # appropriate edge variance.
            leaf_id = leaf.leaf_id
            var_error = error_variances[leaf_id]
            edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            
            if(var_i + var_j <= 0):
                print(i,j,k, var_i, var_j)
            #print(distances)
            #print(i,j,k, var_i, var_j)
            
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.
        diff_top = 0
        sum_of_logs += -diff_top * diff_top / (
            2.0 * edge_variances[-1]
        ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])

        # Return sum of logs of contributions + contribution of ancestor.
        # print(trait_at_node_copy[-1])
        if return_ml_mean == True:
            # For diagnostic purposes--later on always use this estimate!
            return -sum_of_logs, trait_at_node_copy[-1]
        else:
            return -sum_of_logs

    return neg_log_lkl_function
    

def pulse_negloglkl_fun(data, tree, leaves, burst_locations, error_variances=0):
    """Return a function which computes the pulse likelihood of the data,

    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances/const variances."""

    # If measurement error variances are constant throughout...
    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))

    # tree.id is the total number of edges + top - 1
    bursts_array = np.zeros(tree.id + 1)

    # Fill up the burst array to pass to build_cov_matrix
    # bursts_array[i] = # of bursts on the ith edge.
    for burst in burst_locations:
        bursts_array[burst[0]] = burst[1]

    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    
    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    def neg_log_lkl_function(var_p, var_d, return_ml_mean=False):
        # if var_p, var_d < cutoff we return a very high negloglkl
        nm_cutoff = 0

        # For Nelder-Mead, implement a hard boundary.
        if var_p < nm_cutoff:
            return 500 * np.exp(-var_p)
        if var_d < nm_cutoff:
            return 500 * np.exp(-var_d)

        # Define variance matrix for this model
        edge_variances = var_d * bursts_array + var_p * leaf_array

        # Add in measurement errors.
        for leaf in leaves:
            # The errors are numbered according to leaf ID. So we get
            # each leaf's ID, get the error, then append it to the
            # appropriate edge variance.
            leaf_id = leaf.leaf_id
            var_error = error_variances[leaf_id]
            edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.
        diff_top = 0
        sum_of_logs += -diff_top * diff_top / (
            2.0 * edge_variances[-1]
        ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])

        # Return sum of logs of contributions + contribution of ancestor.
        # print(trait_at_node_copy[-1])
        if return_ml_mean == True:
            # For diagnostic purposes--later on always use this estimate!
            return -sum_of_logs, trait_at_node_copy[-1]
        else:
            return -sum_of_logs

    return neg_log_lkl_function



def pulse_negloglkl_fun_withmu(data, tree, leaves, burst_locations, error_variances=0):
    """Return a function which computes the pulse likelihood of the data,

    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances/const variances."""

    # If measurement error variances are constant throughout...
    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))

    # tree.id is the total number of edges + top - 1
    bursts_array = np.zeros(tree.id + 1)

    # Fill up the burst array to pass to build_cov_matrix
    # bursts_array[i] = # of bursts on the ith edge.
    for burst in burst_locations:
        bursts_array[burst[0]] = burst[1]

    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    
    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    def neg_log_lkl_function(var_p, var_d, mu, return_ml_mean=False):
        # if var_p, var_d < cutoff we return a very high negloglkl
        nm_cutoff = 0

        # For Nelder-Mead, implement a hard boundary.
        if var_p < nm_cutoff:
            return 500 * np.exp(-var_p)
        if var_d < nm_cutoff:
            return 500 * np.exp(-var_d)

        # Define variance matrix for this model
        edge_variances = var_d * bursts_array + var_p * leaf_array

        # Add in measurement errors.
        for leaf in leaves:
            # The errors are numbered according to leaf ID. So we get
            # each leaf's ID, get the error, then append it to the
            # appropriate edge variance.
            leaf_id = leaf.leaf_id
            var_error = error_variances[leaf_id]
            edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.
        diff_top = trait_at_node_copy[-1] - mu
        sum_of_logs += -diff_top * diff_top / (
            2.0 * edge_variances[-1]
        ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])

        # Return sum of logs of contributions + contribution of ancestor.
        # print(trait_at_node_copy[-1])
        return -sum_of_logs

    return neg_log_lkl_function



def bm_negloglkl_fun(data, tree, leaves, distances, error_variances=0):
    """Return a function which computes the BM likelihood of the data.

    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances/const variances.
    """

    # If measurement error variances are constant throughout...
    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))



    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    
    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    # No BM variance at the top--but need to add a virtual 0-long edge.
    distances = np.append(distances, [0.])

    def neg_log_lkl_function(var_p, var_bm, return_ml_mean=False):
        # if var_p, var_d < cutoff we return a very high negloglkl
        nm_cutoff = 0

        # For Nelder-Mead, implement a hard boundary.
        if var_p < nm_cutoff:
            return 500 * np.exp(-var_p)
        if var_bm < nm_cutoff:
            return 500 * np.exp(-var_bm)


        # Define variance matrix for this model
        edge_variances = var_p * leaf_array + var_bm * distances

        # Add in measurement errors.
        for leaf in leaves:
            # The errors are numbered according to leaf ID. So we get
            # each leaf's ID, get the error, then append it to the
            # appropriate edge variance.
            leaf_id = leaf.leaf_id
            var_error = error_variances[leaf_id]
            edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.
        diff_top = 0
        sum_of_logs += -diff_top * diff_top / (
            2.0 * edge_variances[-1]
        ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])

        # Return sum of logs of contributions + contribution of ancestor.
        # print(trait_at_node_copy[-1])
        if return_ml_mean == True:
            # For diagnostic purposes--later on always use this estimate!
            return -sum_of_logs, trait_at_node_copy[-1]
        else:
            return -sum_of_logs

    return neg_log_lkl_function

def bm_negloglkl_fun_withmu(data, tree, leaves, distances, error_variances=0):
    """Return a function which computes the BM likelihood of the data.

    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances/const variances.
    """

    # If measurement error variances are constant throughout...
    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))



    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    
    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    # No BM variance at the top--but need to add a virtual 0-long edge.
    distances = np.append(distances, [0.])

    def neg_log_lkl_function(var_p, var_bm, mu, return_ml_mean=False):
        # if var_p, var_d < cutoff we return a very high negloglkl
        nm_cutoff = 0

        # For Nelder-Mead, implement a hard boundary.
        if var_p < nm_cutoff:
            return 500 * np.exp(-var_p)
        if var_bm < nm_cutoff:
            return 500 * np.exp(-var_bm)


        # Define variance matrix for this model
        edge_variances = var_p * leaf_array + var_bm * distances

        # Add in measurement errors.
        for leaf in leaves:
            # The errors are numbered according to leaf ID. So we get
            # each leaf's ID, get the error, then append it to the
            # appropriate edge variance.
            leaf_id = leaf.leaf_id
            var_error = error_variances[leaf_id]
            edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.
        diff_top = trait_at_node_copy[-1] - mu
        sum_of_logs += -diff_top * diff_top / (
            2.0 * edge_variances[-1]
        ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])
        
        return -sum_of_logs

    return neg_log_lkl_function

def bm_mle(data, tree, leaves, distances, error_variances=0):
    """Finds MLE for var_p and var_bm. Returns MLEs, likelihood, AICc."""
    # normalize the data
    data = np.array(data)
    data_mean = data.mean()
    scaling = max(data) - min(data)

    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))
    
    data = (data - data_mean)/scaling
    #data = (data - data.min())/scaling
    errors = np.array(error_variances) / (scaling*scaling)
    
    # now minimize it
    f = bm_negloglkl_fun(data, tree, leaves, distances, error_variances=errors)
    res = brute(lambda x:f(x[0], x[1]), [[1e-30,1], [0,4]], Ns=180, full_output=1)

    # find standard errors
    #_f = bm_negloglkl_fun_withmu(data, tree, leaves, distances, errors)
    #hess = numerical_hessian(lambda x:f(x[0], x[1], x[2]), [res[0][0], res[0][1], res[1]], eps=0.1)
    #bm_vars = np.array(np.linalg.inv(hess).diagonal())
    #bm_vars[0] = scaling*scaling*scaling*scaling*bm_vars[0]
    #bm_vars[1] = scaling*scaling*scaling*scaling*bm_vars[1]
    #bm_vars[2] = scaling*scaling*bm_vars[2]
    

    
    rescaled_mean = scaling*f(res[0][0], res[0][1], True)[1] + data_mean
    
    k = 3 # num. of parameters
    n_data_points = len(data)
    neglogL = res[1] + len(data)*np.log(scaling)
    aic = 2 * k + 2 * neglogL
    aic_c = aic + 2 * k * (k + 1) / (n_data_points - k - 1)
    bic = np.log(len(data))*k + 2*neglogL
    
    
    return [[scaling*scaling*res[0][0], scaling*scaling*res[0][1], rescaled_mean],
            np.exp(-res[1])/(scaling**len(data)),
            aic_c
           ]
    

def pulse_config_mle(data, tree, leaves, pulse_config, error_variances=0, pois=False):
    # normalize the data
    data = np.array(data)
    data_mean = data.mean()
    scaling = max(data) - min(data)

    if isinstance(error_variances, Number):
        error_variances = error_variances * np.ones(len(data))
    
    data = (data - data_mean)/scaling
    errors = np.array(error_variances) / (scaling*scaling)

    # find var_p
    f = _neglkl_marginal_var_p_function(pulse_config, tree, data, errors)
    var_p_optimized = minimize_scalar(f, bounds=[0,1], method='Bounded').x

    #estimate var_d
    _g = pulse_negloglkl_fun(
        data,
        tree,
        leaves,
        pulse_config,
        error_variances=errors
    )
    
    # negloglkl function with fixed var_p
    _conditioned_lkl_vard = lambda x:_g(var_p_optimized,x)
    # find good guess for var_d
    res = minimize_scalar(_conditioned_lkl_vard, bounds=[0,4], method='Bounded')

    # now run gradient descent to find MLEs for var_p and var_d
    res = minimize(lambda x:_g(x[0], x[1]), [var_p_optimized, res.x], bounds=[[1e-20,1],[1e-20,4]])
    
    rescaled_mean = scaling*_g(res.x[0], res.x[1], True)[1] + data_mean

    
    if pois==True:
        total_time = 0
        for distance in distances:
            total_time += distance
            
    
    
    k = 3 # num. of parameters
    
    # add up pulses
    for pulse in pulse_config:
        k += pulse[1]
    
    n_data_points = len(data)
    neglogL = res.fun + len(data)*np.log(scaling)
    aic = 2*k + 2*neglogL
    aic_c = aic + 2*k*(k + 1)/(n_data_points - k - 1)
    #bic = np.log(len(data))*k + 2*neglogL
    
    return [[scaling*scaling*res.x[0], scaling*scaling*res.x[1], rescaled_mean],
            np.exp(-res.fun)/(scaling**len(data)),
            pulse_config,
            aic_c,
           ]

def pulse_iterate_configs(data, tree, leaves, error_variances=0, max_n = 2, min_n = 0, n_processes=-1):
    with Parallel(n_processes) as parallel:
        configurations = parallel(
            delayed(pulse_config_mle)(
                data, tree, leaves, pulse_config, error_variances
            )
            for pulse_config in burst_configs(tree.id, max_n, min_bursts=min_n)
        )

    # Sort the list (minimal AIC_c first)
    configurations.sort(key=lambda x: x[3])
    
    return configurations




def configuration_likelihood_neg_log_lkl_fun(model, *args, **kwargs):
    """Return a function which computes the likelihood of the data, given model.

    model = Models.PULSE/PULSE_MU/PULSE_SEPVARP, PULSE_BM, or BM
    args = data, tree, leaves, burst_locations (pulse)
         = data, tree, leaves, distances, burst_locations (pulsebm)
         = data, tree, leaves, distances (BM)
    
    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances."""
    # data, tree, leaves, distances, burst_locations, model=Models.PULSE
    # ):
    assert isinstance(model, Models), "No model specified!"
    # Depending on model, we either did or did not get a burst_locations
    if model in {Models.PULSE, Models.PULSE_MU, Models.PULSE_SEPVARP}:
        data, tree, leaves, burst_locations = args
        with_pulses = True
    elif model == Models.PULSE_BM:
        data, tree, leaves, distances, burst_locations = args
        with_pulses = True
    elif model == Models.BM:
        data, tree, leaves, distances = args
        with_pulses = False
    else:
        raise ValueError("Unsupported model!")

    # This is a bit of a hack; perhaps model should be a keyword?
    include_errors = False
    if "error_variances" in kwargs:
        error_variances = kwargs["error_variances"]
        include_errors = True

    if with_pulses:
        # tree.id is the total number of edges + top - 1
        bursts_array = np.zeros(tree.id + 1)

        # Fill up the burst array to pass to build_cov_matrix
        # bursts_array[i] = # of bursts on the ith edge.
        for burst in burst_locations:
            bursts_array[burst[0]] = burst[1]

    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    def neg_log_lkl_function(args, return_ml_mean=False):
        # Depending on model, there are different parameters/number of params!
        nm_cutoff = 0
        if model == Models.PULSE:
            mean_value, var_p, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            # Define variance matrix for this model
            edge_variances = var_d * bursts_array + var_p * leaf_array
        if model == Models.PULSE_MU:
            var_p, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            # Define variance matrix for this model
            edge_variances = var_d * bursts_array + var_p * leaf_array
        elif model == Models.PULSE_SEPVARP:
            mean_value, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            estimated_var_p, estimated_var_var_p = pooled_var_p(
                burst_locations, tree, data
            )
            var_p = estimated_var_p
            edge_variances = var_d * bursts_array + var_p * leaf_array
        elif model == Models.PULSE_BM:
            mean_value, var_p, var_d, var_bm = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)
            if var_bm < nm_cutoff:
                return 500 * np.exp(-var_bm)

            distances2 = distances.copy()
            distances2.append(0)
            distances2 = np.array(distances2)
            # Define variance matrix for this model
            edge_variances = (
                var_d * bursts_array + var_p * leaf_array + var_bm * distances2
            )
        elif model == Models.BM:
            mean_value, var_p, var_bm = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_bm < nm_cutoff:
                return 500 * np.exp(-var_bm)

            distances2 = distances.copy()
            distances2.append(0)
            distances2 = np.array(distances2)
            # Define variance matrix for this model
            edge_variances = var_p * leaf_array + var_bm * distances2

        # Add in measurement errors, if we want them in.
        if include_errors:
            for leaf in leaves:
                # The errors are numbered according to leaf ID. So we get
                # each leaf's ID, get the error, then append it to the
                # appropriate edge variance.
                leaf_id = leaf.leaf_id
                var_error = error_variances[leaf_id]
                edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

        # Contribution of most common ancestor is different.

        if model == Models.PULSE_MU:
            diff_top = 0
        else:
            diff_top = trait_at_node_copy[-1] - mean_value
        # MLE will be s.t. diff_top = 0 but we'll leave this here for now.
        try:
            sum_of_logs += -diff_top * diff_top / (
                2.0 * edge_variances[-1]
            ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])
        except RuntimeWarning:
            print("Edge_Variances[-1]={}".format(edge_variances[-1]))

        # Return sum of logs of contributions + contribution of ancestor.
        # print(trait_at_node_copy[-1])
        if return_ml_mean == True:
            # For diagnostic purposes--later on always use this estimate!
            return -sum_of_logs, trait_at_node_copy[-1]
        else:
            return -sum_of_logs

    return neg_log_lkl_function


def configuration_likelihood_neg_log_lkl_fun_2(model, *args, **kwargs):
    """Return a function which computes the likelihood of the data, given model.
    
    This function always chooses the MLE mean! Therefore mu can't be set.

    model = Models.PULSE/PULSE_MU/PULSE_SEPVARP, PULSE_BM, or BM
    args = data, tree, leaves, burst_locations (pulse)
         = data, tree, leaves, distances, burst_locations (pulsebm)
         = data, tree, leaves, distances (BM)
    
    If there are VARIANCES on the data measurements, specify them by
    a keyword: error_variances = vector_of_variances."""
    # data, tree, leaves, distances, burst_locations, model=Models.PULSE
    # ):
    assert isinstance(model, Models), "No model specified!"
    # Depending on model, we either did or did not get a burst_locations
    if model in {Models.PULSE, Models.PULSE_MU, Models.PULSE_SEPVARP}:
        data, tree, leaves, burst_locations = args
        with_pulses = True
    elif model == Models.PULSE_BM:
        data, tree, leaves, distances, burst_locations = args
        with_pulses = True
    elif model == Models.BM:
        data, tree, leaves, distances = args
        with_pulses = False
    else:
        raise ValueError("Unsupported model!")

    # This is a bit of a hack; perhaps model should be a keyword?
    include_errors = False
    if "error_variances" in kwargs:
        error_variances = kwargs["error_variances"]
        include_errors = True

    if with_pulses:
        # tree.id is the total number of edges + top - 1
        bursts_array = np.zeros(tree.id + 1)

        # Fill up the burst array to pass to build_cov_matrix
        # bursts_array[i] = # of bursts on the ith edge.
        for burst in burst_locations:
            bursts_array[burst[0]] = burst[1]

    # Set leaf trait values; rest will be updated later.
    trait_at_node = np.zeros(tree.id + 1)
    # 1 if edge with given id is a leaf.
    leaf_array = np.zeros(tree.id + 1)

    # Fill up above two arrays.
    for i in range(len(leaves)):
        # So far we only know traits at leaves, but we'll populate rest
        # later.
        trait_at_node[leaves[i].id] = data[i]
        # Indicator of whether an edge is a leaf of not.
        leaf_array[leaves[i].id] = 1.0

    # Write down order in which to compute the contrasts (i.e. postorder of
    # edges, without leaves), and the left and right descendants of each edge.
    nodes_to_compute = []

    for node in tree.traverse("postorder"):
        if node.is_leaf() == False:
            childs = node.get_children()
            # Note: Here we are making the assumption that we have a binary tree!
            nodes_to_compute.append([node.id, childs[0].id, childs[1].id])

    # The likelihood of the given configuration, given mu, var_p, and var_d.
    # This is what we will maximize and return max as "configuration
    # likelihood".
    def neg_log_lkl_function(args, return_ml_mean=False):
        # Depending on model, there are different parameters/number of params!
        nm_cutoff = 0
        if model == Models.PULSE:
            mean_value, var_p, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            # Define variance matrix for this model
            edge_variances = var_d * bursts_array + var_p * leaf_array
        if model == Models.PULSE_MU:
            var_p, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            # Define variance matrix for this model
            edge_variances = var_d * bursts_array + var_p * leaf_array
        elif model == Models.PULSE_SEPVARP:
            mean_value, var_d = args

            # For Nelder-Mead, implement a hard boundary.
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)

            estimated_var_p, estimated_var_var_p = pooled_var_p(
                burst_locations, tree, data
            )
            var_p = estimated_var_p
            edge_variances = var_d * bursts_array + var_p * leaf_array
        elif model == Models.PULSE_BM:
            mean_value, var_p, var_d, var_bm = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_d < nm_cutoff:
                return 500 * np.exp(-var_d)
            if var_bm < nm_cutoff:
                return 500 * np.exp(-var_bm)

            distances2 = distances.copy()
            distances2.append(0)
            distances2 = np.array(distances2)
            # Define variance matrix for this model
            edge_variances = (
                var_d * bursts_array + var_p * leaf_array + var_bm * distances2
            )
        elif model == Models.BM:
            mean_value, var_p, var_bm = args

            # For Nelder-Mead, implement a hard boundary.
            if var_p < nm_cutoff:
                return 500 * np.exp(-var_p)
            if var_bm < nm_cutoff:
                return 500 * np.exp(-var_bm)

            distances2 = distances.copy()
            distances2.append(0)
            distances2 = np.array(distances2)
            # Define variance matrix for this model
            edge_variances = var_p * leaf_array + var_bm * distances2

        # Add in measurement errors, if we want them in.
        if include_errors:
            for leaf in leaves:
                # The errors are numbered according to leaf ID. So we get
                # each leaf's ID, get the error, then append it to the
                # appropriate edge variance.
                leaf_id = leaf.leaf_id
                var_error = error_variances[leaf_id]
                edge_variances[leaf.id] += var_error

        trait_at_node_copy = trait_at_node.copy()

        sum_of_logs = 0.0

        # k is index of parent; i, j - index of children.
        for triplet in nodes_to_compute:
            k = triplet[0]
            i = triplet[1]
            j = triplet[2]
            var_i = edge_variances[i]
            var_j = edge_variances[j]
            trait_i = trait_at_node_copy[i]
            trait_j = trait_at_node_copy[j]

            # Update variance for pruned tree.
            # if (var_i + var_j < 1e-60):
            #    return 1000
            # edge_variances[k] += var_i * var_j / (var_i + var_j)
            edge_variances[k] += var_i * var_j / (var_i + var_j)

            # Estimate trait at node k.
            trait_at_node_copy[k] = (trait_j * var_i + trait_i * var_j) / (
                var_i + var_j
            )

            diff_k = trait_at_node_copy[i] - trait_at_node_copy[j]

            sum_of_logs += -diff_k * diff_k / (2 * (var_i + var_j)) - 0.5 * np.log(
                2 * np.pi * (var_i + var_j)
            )

            # We choose diff_top = 0.
            # sum_of_logs += -diff_top * diff_top / (
            #    2.0 * edge_variances[-1]
            # ) + -0.5 * np.log(2.0 * np.pi * edge_variances[-1])
            sum_of_logs += -0.5 * np.log(2.0 * np.pi * edge_variances[-1])

        if return_ml_mean == True:
            # For diagnostic purposes--later on always use this estimate!
            return -sum_of_logs, trait_at_node_copy[-1]
        else:
            return -sum_of_logs

    return neg_log_lkl_function


def configuration_likelihood_unified_2(model, *args, **kwargs):
    """Compute the MLE and relevant info by specifying the model and data.

    model = Models.PULSE, PULSE_BM, or BM
    args = data, tree, leaves, burst_locations (pulse)
         = data, tree, leaves, distances, burst_locations (pulsebm)
         = data, tree, leaves, distances (BM)
    
    burst_locations = e.g. [(19,1),(22,1),(24,1)]

    If there are VARIANCES (measuremt errors) on the data measurements,
    speciy them by a keyword: error_variances = vector_of_variances.

    Note that the variances on the fitted parameters are computed from
    the APPROXIMATE inverted Hessian (i.e. a second derivative is never
    computed!). To compute a more exact number, use included routine to
    compute a central difference approximation for the Hessian,
    then invert it."""
    # data, tree, leaves, distances, burst_locations, model=Models.PULSE
    # ):
    assert isinstance(model, Models), "No model specified!"
    # Depending on model, we either did or did not get a burst_locations
    if model in {Models.PULSE}:
        data, tree, leaves, burst_locations = args
        with_pulses = True
    elif model == Models.PULSE_BM:
        data, tree, leaves, distances, burst_locations = args
        with_pulses = True
    elif model == Models.BM:
        data, tree, leaves, distances = args
        with_pulses = False
    else:
        raise ValueError("Unsupported model!")

    # This is a bit of a hack; perhaps model should be a keyword?
    include_errors = False
    if "error_variances" in kwargs:
        error_variances = kwargs["error_variances"]
        include_errors = True

    # The likelihood of the given configuration, given mu, var_p, and others.
    # This is what we will minimize to get a (negative log of) "configuration
    # likelihood".
    neg_log_lkl_function = configuration_likelihood_neg_log_lkl_fun(
        model, *args, **kwargs
    )

    # Now minimize above-defined function, We need different bounds and initial
    # guesses for each model.
    # Originally we used a finite domain, but it doesn't seem needed. However!
    # We stil put eps > 0 so that we don't test the case where var_p=var_d=0,
    # since then we can potentially divide by zero.

    if model == Models.PULSE:
        variable_labels = ["var_p", "var_d"]
        # We're assuming the data is normalized to 1...
        initial_guess = [
            pooled_var_p(burst_locations, tree, data)[
                0
            ],  # get only estimate, not error in estimate
            sigma_d_estimate(data),
        ]
    elif model == Models.PULSE_BM:
        variable_labels = ["var_p", "var_d", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [
            sigma_p_estimate(data),
            sigma_d_estimate(data) / 2.0,
            sigma_d_estimate(data) / (4 * height),
        ]
    elif model == Models.BM:
        variable_labels = ["var_p", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [sigma_p_estimate(data), sigma_d_estimate(data) / (2 * height)]
    # Set the bounds in terms of the initial guesses.
    # Different variance parameters
    bnds = []
    for component in initial_guess:
        bnds.append(component / 100.0, 3 * component)
        # bnds.append(component / 5.0, 3*component)

    # And now we minimize!
    initial_guess = [x for x in initial_guess]
    if "method" in kwargs:
        if kwargs["method"] == "Nelder-Mead":
            res = minimize(
                neg_log_lkl_function, initial_guess, jac=False, method="Nelder-Mead"
            )
        if kwargs["method"] == "basinhopping":
            # max size of hop is half the length of the initial guess length--an ad hoc choice.
            max_hop = np.sqrt((np.array(initial_guess) ** 2).sum()) / 5
            res = basinhopping(
                neg_log_lkl_function, initial_guess, niter=20, stepsize=max_hop
            )
        if kwargs["method"] == "brute":
            # res = brute(neg_log_lkl_function, bnds, Ns=20)
            res = brute(
                neg_log_lkl_function, bnds, Ns=4, finish=minimize, args={"bounds": bnds}
            )
        else:
            raise ValueError("{} is not a recognized method!".format(kwargs["method"]))
    else:
        res = brute(
            neg_log_lkl_function,
            bnds,
            Ns=15,
            finish=minimize,
            args={"bounds": bnds}
            # res = minimize(
            #    neg_log_lkl_function,
            #    initial_guess,
            #    jac=False,
            #    bounds=bnds,
            #    options={"gtol": 1e-50},
        )

    if "method" in kwargs and kwargs["method"] == "brute":
        ml_parameters = res
        min_neg_log_lkl = neg_log_lkl_function(res)
    else:
        ml_parameters = res.x
        min_neg_log_lkl = res.fun

    # Inverse of the Hessian of -log(likelihood) is the asymptotic cov matrix.
    # In principle we should probably compute all of this exactly, but the
    # approximate hessian inverse seems good enough.
    # parameter_variances = []
    # for i in range(len(res.x)):
    #    parameter_variances.append(res.hess_inv[i][i])

    # return [max_lkl, res.x, list(res.hess_inv.diagonal()), burst_locations]
    # fitted parameters

    fit_results = OrderedDict(zip(variable_labels, ml_parameters))

    # variances on fitted parameters are diagonal terms of observed Fisher
    # matrix
    # This actuall returns LbfgsInvHessProduct, which we convert to matrix.
    # obs_Fisher_matrix = np.eye(len(initial_guess))
    # obs_Fisher_matrix = res.hess_inv.todense()
    # variance_labels = ["var_of_" + x for x in variable_labels]
    # fit_variances = OrderedDict(zip(variance_labels, obs_Fisher_matrix.diagonal()))

    # Here we compute the AIC and AICc scores.

    n_data_points = len(data)
    n_params = len(initial_guess)

    # PULSE_SEPVARP computes var_p separately, but that still counts as a parameter!
    if model == Models.PULSE_SEPVARP:
        n_params += 1
    elif model == Models.PULSE_MU:
        n_params += 1
    elif model == Models.PULSE:
        pass
    elif model == Models.PULSE_BM:
        pass
    elif model == Models.BM:
        pass
    else:
        # This is to avoid incorrectly computed AICc bugs, i.e.
        # we must decide the number of parameters explicitly.
        n_params = 100000

    # We count each pulse position as a parameter.
    if with_pulses:
        n_pulses = 0

        # Add up number of pulses in each branch.
        # Note len(burst_locations) won't work--might undercount e.g. (1,5)!
        for pulse in burst_locations:
            n_pulses += pulse[1]

        n_params += n_pulses

    aic = 2.0 * n_params + 2 * min_neg_log_lkl
    aic_c = aic + 2 * n_params * (n_params + 1) / (n_data_points - n_params - 1)

    # Return max likelihood, given this configuration of bursts.
    max_lkl = np.exp(-min_neg_log_lkl)

    # This is what we'll return
    analysis_labels = [
        "max_lkl",
        "fit_values",
        "initial_guess",
        # "fit_variances_approx",
        "n_parameters",
        "aic",
        "aic_c",
        "minimizer_result",
        "model",
        "used_measurement_variances",
    ]

    analysis_content = [
        max_lkl,
        fit_results,
        initial_guess,
        # fit_variances,
        n_params,
        aic,
        aic_c,
        res,
        model,
        include_errors,
    ]

    # This one only if there are burst locations.
    # Maybe for BM we should return an empty list?
    if with_pulses:
        analysis_labels.append("pulse_positions")
        analysis_content.append(burst_locations)
    if model == Models.PULSE_SEPVARP:
        estimated_var_p, estimated_var_var_p = pooled_var_p(burst_locations, tree, data)
        analysis_labels.append("var_p_regime")
        analysis_content.append(estimated_var_p)
        analysis_labels.append("var_var_p_regime")
        analysis_content.append(estimated_var_var_p)
        # analysis_content['fit_values']['var_p'] = estimated_var_p
        # Convention is that var_d is at the end.
        # analysis_content['fit_values'].move_to_end('var_d')

    assert len(analysis_labels) == len(
        analysis_content
    ), "Labels and content mismatched!"
    return OrderedDict(zip(analysis_labels, analysis_content))


def configuration_likelihood_unified(model, *args, **kwargs):
    """Compute the MLE and relevant info by specifying the model and data.

    model = Models.PULSE, PULSE_BM, or BM
    args = data, tree, leaves, burst_locations (pulse)
         = data, tree, leaves, distances, burst_locations (pulsebm)
         = data, tree, leaves, distances (BM)
    
    burst_locations = e.g. [(19,1),(22,1),(24,1)]

    If there are VARIANCES (measuremt errors) on the data measurements,
    speciy them by a keyword: error_variances = vector_of_variances.

    Note that the variances on the fitted parameters are computed from
    the APPROXIMATE inverted Hessian (i.e. a second derivative is never
    computed!). To compute a more exact number, use included routine to
    compute a central difference approximation for the Hessian,
    then invert it."""
    # data, tree, leaves, distances, burst_locations, model=Models.PULSE
    # ):
    assert isinstance(model, Models), "No model specified!"
    # Depending on model, we either did or did not get a burst_locations
    if model in {Models.PULSE, Models.PULSE_MU, Models.PULSE_SEPVARP}:
        data, tree, leaves, burst_locations = args
        with_pulses = True
    elif model == Models.PULSE_BM:
        data, tree, leaves, distances, burst_locations = args
        with_pulses = True
    elif model == Models.BM:
        data, tree, leaves, distances = args
        with_pulses = False
    else:
        raise ValueError("Unsupported model!")

    # This is a bit of a hack; perhaps model should be a keyword?
    include_errors = False
    if "error_variances" in kwargs:
        error_variances = kwargs["error_variances"]
        include_errors = True

    # The likelihood of the given configuration, given mu, var_p, and others.
    # This is what we will minimize to get a (negative log of) "configuration
    # likelihood".
    neg_log_lkl_function = configuration_likelihood_neg_log_lkl_fun(
        model, *args, **kwargs
    )

    # Now minimize above-defined function, We need different bounds and initial
    # guesses for each model.
    # Originally we used a finite domain, but it doesn't seem needed. However!
    # We stil put eps > 0 so that we don't test the case where var_p=var_d=0,
    # since then we can potentially divide by zero.

    if model == Models.PULSE_MU:
        variable_labels = ["var_p", "var_d"]
        var_p_est = pooled_var_p(burst_locations, tree, data)[0]
        var_d_est = sigma_d_estimate(data)
        initial_guess = [var_p_est, var_d_est]
    elif model == Models.PULSE:
        variable_labels = ["mean", "var_p", "var_d"]
        # We're assuming the data is normalized to 1...
        initial_guess = [
            mean_ancestor_estimate(data),
            pooled_var_p(burst_locations, tree, data)[
                0
            ],  # get only estimate, not error in estimate
            sigma_d_estimate(data),
        ]
        var_p_est = initial_guess[1]
        var_d_est = initial_guess[2]
    elif model == Models.PULSE_SEPVARP:
        variable_labels = ["mean", "var_d"]
        # We're assuming the data is normalized to 1...
        initial_guess = [mean_ancestor_estimate(data), sigma_d_estimate(data)]
    elif model == Models.PULSE_BM:
        variable_labels = ["mean", "var_p", "var_d", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [
            mean_ancestor_estimate(data),
            sigma_p_estimate(data),
            sigma_d_estimate(data) / 2.0,
            sigma_d_estimate(data) / (4 * height),
        ]
    elif model == Models.BM:
        variable_labels = ["mean", "var_p", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [
            mean_ancestor_estimate(data),
            sigma_p_estimate(data),
            sigma_d_estimate(data) / (2 * height),
        ]
    # Set the bounds in terms of the initial guesses.

    # Include the mean or not; offset_mean = 0 means no, 1 yes.
    offset_mean = 0
    bnds = len(initial_guess) * [[0, 0]]
    bnds_brute = len(initial_guess) * [[0, 0]]

    if model != Models.PULSE_MU:
        bnds[0][0] = min(data)
        bnds[0][1] = max(data)
        bnds_brute[0][0] = min(data)
        bnds_brute[0][1] = max(data)
        offset_mean = 1

    # Different variance parameters
    # for i in range(offset_mean, len(bnds)):
    #    bnds[i][0] = 1e-20
    #    bnds[i][1] = 4*(max(data) - min(data))**2

    for i in range(offset_mean, len(bnds)):
        bnds[i][0] = 1e-10
        bnds[i][1] = 1
        bnds_brute[i][0] = 1e-5
        bnds_brute[i][1] = 1

    bnds[offset_mean][0] = 0.5e-10
    bnds[offset_mean][1] = 0.08
    bnds_brute[offset_mean][0] = 0.5e-2
    bnds_brute[offset_mean][1] = 0.01

    # bnds[0][0] = -3.0 * abs(initial_guess[0])
    # bnds[0][1] = 3.0 * abs(initial_guess[0])

    # for i in range(1, 3):
    #    bnds[i][0] = initial_guess[i] / 10.0
    #    bnds[i][1] = initial_guess[i] * 3

    # And now we minimize!
    # initial_guess = [x for x in initial_guess]
    # for i in range(1, len(initial_guess)):
    #    initial_guess[i] = initial_guess[i]
    if "method" in kwargs:
        if kwargs["method"] == "Nelder-Mead":
            res = minimize(
                neg_log_lkl_function, initial_guess, jac=False, method="Nelder-Mead"
            )
        if kwargs["method"] == "basinhopping":
            # max size of hop is half the length of the initial guess length--an ad hoc choice.
            max_hop = np.sqrt((np.array(initial_guess) ** 2).sum()) / 5
            res = basinhopping(
                neg_log_lkl_function, initial_guess, niter=20, stepsize=max_hop
            )
        if kwargs["method"] == "brute":
            res = brute(
                neg_log_lkl_function,
                bnds_brute,
                Ns=10,
                finish=minimize,
                args={"bounds": bnds},
            )
        else:
            raise ValueError("{} is not a recognized method!".format(kwargs["method"]))
    else:
        res = minimize(
            neg_log_lkl_function,
            initial_guess,
            # jac=False,
            bounds=bnds,
            # options={"gtol": 1e-25},
        )

    if "method" in kwargs:
        if kwargs["method"] == "brute":
            ml_parameters = res
            min_neg_log_lkl = neg_log_lkl_function(res)
        else:
            ml_parameters = res.x
            min_neg_log_lkl = res.fun
    else:
        ml_parameters = res.x
        min_neg_log_lkl = res.fun

    # Inverse of the Hessian of -log(likelihood) is the asymptotic cov matrix.
    # In principle we should probably compute all of this exactly, but the
    # approximate hessian inverse seems good enough.
    # parameter_variances = []
    # for i in range(len(res.x)):
    #    parameter_variances.append(res.hess_inv[i][i])

    # return [max_lkl, res.x, list(res.hess_inv.diagonal()), burst_locations]
    # fitted parameters

    fit_results = OrderedDict(zip(variable_labels, ml_parameters))

    # variances on fitted parameters are diagonal terms of observed Fisher
    # matrix
    # This actuall returns LbfgsInvHessProduct, which we convert to matrix.
    # obs_Fisher_matrix = np.eye(len(initial_guess))
    # obs_Fisher_matrix = res.hess_inv.todense()
    # variance_labels = ["var_of_" + x for x in variable_labels]
    # fit_variances = OrderedDict(zip(variance_labels, obs_Fisher_matrix.diagonal()))

    # Here we compute the AIC and AICc scores.

    n_data_points = len(data)
    n_params = len(initial_guess)

    # PULSE_SEPVARP computes var_p separately, but that still counts as a parameter!
    if model == Models.PULSE_SEPVARP:
        n_params += 1
    elif model == Models.PULSE_MU:
        n_params += 1
    elif model == Models.PULSE:
        pass
    elif model == Models.PULSE_BM:
        pass
    elif model == Models.BM:
        pass
    else:
        # This is to avoid incorrectly computed AICc bugs, i.e.
        # we must decide the number of parameters explicitly.
        n_params = 100000
    # We count each pulse position as a parameter.
    if with_pulses:
        n_pulses = 0

        # Add up number of pulses in each branch.
        # Note len(burst_locations) won't work--might undercount e.g. (1,5)!
        for pulse in burst_locations:
            n_pulses += pulse[1]

        n_params += n_pulses

    aic = 2.0 * n_params + 2 * min_neg_log_lkl
    aic_c = aic + 2 * n_params * (n_params + 1) / (n_data_points - n_params - 1)

    # Return max likelihood, given this configuration of bursts.
    max_lkl = np.exp(-min_neg_log_lkl)

    # This is what we'll return
    analysis_labels = [
        "max_lkl",
        "fit_values",
        "initial_guess",
        # "fit_variances_approx",
        "n_parameters",
        "aic",
        "aic_c",
        "minimizer_result",
        "model",
        "used_measurement_variances",
    ]

    analysis_content = [
        max_lkl,
        fit_results,
        initial_guess,
        # fit_variances,
        n_params,
        aic,
        aic_c,
        res,
        model,
        include_errors,
    ]

    # This one only if there are burst locations.
    # Maybe for BM we should return an empty list?
    if with_pulses:
        analysis_labels.append("pulse_positions")
        analysis_content.append(burst_locations)
    if model == Models.PULSE_SEPVARP:
        estimated_var_p, estimated_var_var_p = pooled_var_p(burst_locations, tree, data)
        analysis_labels.append("var_p_regime")
        analysis_content.append(estimated_var_p)
        analysis_labels.append("var_var_p_regime")
        analysis_content.append(estimated_var_var_p)
        # analysis_content['fit_values']['var_p'] = estimated_var_p
        # Convention is that var_d is at the end.
        # analysis_content['fit_values'].move_to_end('var_d')

    assert len(analysis_labels) == len(
        analysis_content
    ), "Labels and content mismatched!"
    return OrderedDict(zip(analysis_labels, analysis_content))


def configuration_likelihood_unified_nomu(model, *args, **kwargs):
    """Compute the MLE and relevant info by specifying the model and data.

    model = Models.PULSE, PULSE_BM, or BM
    args = data, tree, leaves, burst_locations (pulse)
         = data, tree, leaves, distances, burst_locations (pulsebm)
         = data, tree, leaves, distances (BM)
    
    burst_locations = e.g. [(19,1),(22,1),(24,1)]

    If there are VARIANCES (measuremt errors) on the data measurements,
    speciy them by a keyword: error_variances = vector_of_variances.

    Note that the variances on the fitted parameters are computed from
    the APPROXIMATE inverted Hessian (i.e. a second derivative is never
    computed!). To compute a more exact number, use included routine to
    compute a central difference approximation for the Hessian,
    then invert it."""
    # data, tree, leaves, distances, burst_locations, model=Models.PULSE
    # ):
    assert isinstance(model, Models), "No model specified!"
    # Depending on model, we either did or did not get a burst_locations
    if model in {Models.PULSE, Models.PULSE_MU, Models.PULSE_SEPVARP}:
        data, tree, leaves, burst_locations = args
        with_pulses = True
    elif model == Models.PULSE_BM:
        data, tree, leaves, distances, burst_locations = args
        with_pulses = True
    elif model == Models.BM:
        data, tree, leaves, distances = args
        with_pulses = False
    else:
        raise ValueError("Unsupported model!")

    # This is a bit of a hack; perhaps model should be a keyword?
    include_errors = False
    if "error_variances" in kwargs:
        error_variances = kwargs["error_variances"]
        include_errors = True

    # The likelihood of the given configuration, given mu, var_p, and others.
    # This is what we will minimize to get a (negative log of) "configuration
    # likelihood".
    neg_log_lkl_function = configuration_likelihood_neg_log_lkl_fun(
        model, *args, **kwargs
    )

    # Now minimize above-defined function, We need different bounds and initial
    # guesses for each model.
    # Originally we used a finite domain, but it doesn't seem needed. However!
    # We stil put eps > 0 so that we don't test the case where var_p=var_d=0,
    # since then we can potentially divide by zero.

    if model == Models.PULSE_MU:
        variable_labels = ["var_p", "var_d"]
        var_p_est = pooled_var_p(burst_locations, tree, data)[0]
        var_d_est = sigma_d_estimate(data)
        initial_guess = [var_p_est, var_d_est]
    if model == Models.PULSE:
        variable_labels = ["mean", "var_p", "var_d"]
        # We're assuming the data is normalized to 1...
        initial_guess = [
            mean_ancestor_estimate(data),
            pooled_var_p(burst_locations, tree, data)[
                0
            ],  # get only estimate, not error in estimate
            sigma_d_estimate(data),
        ]
    elif model == Models.PULSE_SEPVARP:
        variable_labels = ["mean", "var_d"]
        # We're assuming the data is normalized to 1...
        initial_guess = [mean_ancestor_estimate(data), sigma_d_estimate(data)]
    elif model == Models.PULSE_BM:
        variable_labels = ["mean", "var_p", "var_d", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [
            mean_ancestor_estimate(data),
            sigma_p_estimate(data),
            sigma_d_estimate(data) / 2.0,
            sigma_d_estimate(data) / (4 * height),
        ]
    elif model == Models.BM:
        variable_labels = ["mean", "var_p", "var_bm"]
        # We assume the tree is ultrametric, so height is just distance from
        # top to the first leag. We do this to estimate var_bm.
        height = tree.get_distance(leaves[0])
        # We assume sigma_bm^2 is inversely proportional to tree height;
        # we also assume the variance due to displacement and BM is approx.
        # divided in a ratio if 1:2. (just a very rough guess!)
        initial_guess = [
            mean_ancestor_estimate(data),
            sigma_p_estimate(data),
            sigma_d_estimate(data) / (2 * height),
        ]
    # Set the bounds in terms of the initial guesses.

    # Mean
    bnds = len(initial_guess) * [[0, 0]]
    bnds[0][0] = -3.0 * abs(initial_guess[0])
    bnds[0][1] = 3.0 * abs(initial_guess[0])

    # Different variance parameters
    for i in range(1, len(bnds)):
        bnds[i][0] = initial_guess[i] / 100.0
        bnds[i][1] = initial_guess[i] * 10
    # bnds[0][0] = -3.0 * abs(initial_guess[0])
    # bnds[0][1] = 3.0 * abs(initial_guess[0])

    # for i in range(1, 3):
    #    bnds[i][0] = initial_guess[i] / 10.0
    #    bnds[i][1] = initial_guess[i] * 3

    # And now we minimize!
    initial_guess = [x for x in initial_guess]
    if "method" in kwargs:
        if kwargs["method"] == "Nelder-Mead":
            res = minimize(
                neg_log_lkl_function, initial_guess, jac=False, method="Nelder-Mead"
            )
        if kwargs["method"] == "basinhopping":
            # max size of hop is half the length of the initial guess length--an ad hoc choice.
            max_hop = np.sqrt((np.array(initial_guess) ** 2).sum()) / 5
            res = basinhopping(
                neg_log_lkl_function, initial_guess, niter=20, stepsize=max_hop
            )
        if kwargs["method"] == "brute":
            res = brute(
                neg_log_lkl_function,
                [
                    [var_p_est / 10000, 5 * var_p_est],
                    [var_d_est / 10000, 5 * var_d_est],
                ],
            )
        else:
            raise ValueError("{} is not a recognized method!".format(kwargs["method"]))
    else:
        res = minimize(
            neg_log_lkl_function,
            initial_guess,
            jac=False,
            bounds=bnds,
            options={"gtol": 1e-25},
        )

    if "method" in kwargs and kwargs["method"] == "brute":
        ml_parameters = res
        min_neg_log_lkl = neg_log_lkl_function(res)
    else:
        ml_parameters = res.x
        min_neg_log_lkl = res.fun

    # Inverse of the Hessian of -log(likelihood) is the asymptotic cov matrix.
    # In principle we should probably compute all of this exactly, but the
    # approximate hessian inverse seems good enough.
    # parameter_variances = []
    # for i in range(len(res.x)):
    #    parameter_variances.append(res.hess_inv[i][i])

    # return [max_lkl, res.x, list(res.hess_inv.diagonal()), burst_locations]
    # fitted parameters

    fit_results = OrderedDict(zip(variable_labels, ml_parameters))

    # variances on fitted parameters are diagonal terms of observed Fisher
    # matrix
    # This actuall returns LbfgsInvHessProduct, which we convert to matrix.
    # obs_Fisher_matrix = np.eye(len(initial_guess))
    # obs_Fisher_matrix = res.hess_inv.todense()
    # variance_labels = ["var_of_" + x for x in variable_labels]
    # fit_variances = OrderedDict(zip(variance_labels, obs_Fisher_matrix.diagonal()))

    # Here we compute the AIC and AICc scores.

    n_data_points = len(data)
    n_params = len(initial_guess)

    # PULSE_SEPVARP computes var_p separately, but that still counts as a parameter!
    if model == Models.PULSE_SEPVARP:
        n_params += 1

    # We count each pulse position as a parameter.
    if with_pulses:
        n_pulses = 0

        # Add up number of pulses in each branch.
        # Note len(burst_locations) won't work--might undercount e.g. (1,5)!
        for pulse in burst_locations:
            n_pulses += pulse[1]

        n_params += n_pulses

    aic = 2.0 * n_params + 2 * min_neg_log_lkl
    aic_c = aic + 2 * n_params * (n_params + 1) / (n_data_points - n_params - 1)

    # Return max likelihood, given this configuration of bursts.
    max_lkl = np.exp(-min_neg_log_lkl)

    # This is what we'll return
    analysis_labels = [
        "max_lkl",
        "fit_values",
        "initial_guess",
        # "fit_variances_approx",
        "n_parameters",
        "aic",
        "aic_c",
        "minimizer_result",
        "model",
        "used_measurement_variances",
    ]

    analysis_content = [
        max_lkl,
        fit_results,
        initial_guess,
        # fit_variances,
        n_params,
        aic,
        aic_c,
        res,
        model,
        include_errors,
    ]

    # This one only if there are burst locations.
    # Maybe for BM we should return an empty list?
    if with_pulses:
        analysis_labels.append("pulse_positions")
        analysis_content.append(burst_locations)
    if model == Models.PULSE_SEPVARP:
        estimated_var_p, estimated_var_var_p = pooled_var_p(burst_locations, tree, data)
        analysis_labels.append("var_p_regime")
        analysis_content.append(estimated_var_p)
        analysis_labels.append("var_var_p_regime")
        analysis_content.append(estimated_var_var_p)
        # analysis_content['fit_values']['var_p'] = estimated_var_p
        # Convention is that var_d is at the end.
        # analysis_content['fit_values'].move_to_end('var_d')

    assert len(analysis_labels) == len(
        analysis_content
    ), "Labels and content mismatched!"
    return OrderedDict(zip(analysis_labels, analysis_content))


def numerical_hessian(f, vec, eps=1e-2, fwd=True):
    """Compute the Hessian using _forward_ diffrences, because var_p might be close to 0.
    fwd=True for forward difference, fwd=False for central difference."""
    vec = np.array(vec)
    dim = len(vec)
    hess = np.zeros((dim, dim))
    
    if fwd==True:
        a = 2.0
        b = 1.0
    else:
        a = 1.
        b = -1
    
    for i in range(dim):
        for j in range(dim):
            perturb_vec_i = np.zeros(dim)
            perturb_vec_i[i] = eps
            perturb_vec_j = np.zeros(dim)
            perturb_vec_j[j] = eps

            if i != j:
                #vec_pp = vec + a*perturb_vec_i + a*perturb_vec_j
                #vec_mm = vec + b*perturb_vec_i + b*perturb_vec_j
                #vec_pm = vec + a*perturb_vec_i + b*perturb_vec_j
                #vec_mp = vec + b*perturb_vec_i + a*perturb_vec_j
                
                vec_pp = vec + 2*perturb_vec_i + 2*perturb_vec_j
                vec_mm = vec + perturb_vec_i + perturb_vec_j
                vec_pm = vec + 2*perturb_vec_i + perturb_vec_j
                vec_mp = vec + perturb_vec_i + 2*perturb_vec_j

                f_pp = f(vec_pp)
                f_mm = f(vec_mm)
                f_pm = f(vec_pm)
                f_mp = f(vec_mp)
                hess[i][j] = (f_mm + f_pp - f_pm - f_mp) / (4 * eps * eps)
            else:
                #vec_p = vec + 2*perturb_vec_i
                #vec_m = vec + perturb_vec_i
                vec_p = vec + 2*perturb_vec_i
                vec_m = vec + perturb_vec_i
                f_p = f(vec_p)
                f_m = f(vec_m)
                f_0 = f(vec)
                hess[i][i] = (f_p - 2 * f_0 + f_m) / (eps * eps)

    return hess


def _check_all_configurations(
    model,
    treefile,
    datafile,
    with_meas_errors=True,
    max_n=2,
    min_n=0,
    n_processes=-1,
    eps=1e-7,
    method=None,
    rescale_factor="minmax",
):
    """Compute the likelihoods and other data up to max_n pulses.
    
    Note: Data will be rescaled by its mean!
    Set min_n=max_n to ONLY do max_n pulses. Set with_meas_errors=False to note load
    measurement error variances. Set n_processes=-1 for max number of processes, or
    to n_processes=K to specify K parallel processes. Variances in fit parameters
    for best config are computed by inverting finite-diff Hessian with step size eps."""

    # For recording when analysis started
    start_datetime = datetime.now()

    loaded_files = load_treefile_datafile(
        treefile, datafile, return_errors=with_meas_errors
    )
    tree = loaded_files[0]
    data = np.array(loaded_files[1])
    n_leaves, leaves = number_leaves(tree)
    n_nodes, nodes, distances = number_edges(tree)

    # Now let's normalize the data
    # Shift data by mean in min_max rescaling.
    shift = 0

    if rescale_factor == "mean":
        rescale_factor = data.sum() / len(data)
    if rescale_factor == "minmax":
        rescale_factor = data.max() - data.min()
        shift = data.sum() / len(data)

    data = (data - shift) / rescale_factor

    kwargs = {}
    scaled_meas_err = False
    if with_meas_errors:
        scaled_meas_err = np.array(loaded_files[2]) / (rescale_factor * rescale_factor)
        kwargs["error_variances"] = scaled_meas_err

    if method != None:
        kwargs["method"] = method

    if model == Models.BM:
        configurations = [
            configuration_likelihood_unified(
                Models.BM, data, tree, leaves, distances, **kwargs
            )
        ]

    # We'll need to append the pulse positions to args!
    if model in {Models.PULSE, Models.PULSE_SEPVARP, Models.PULSE_MU}:
        partial_args = (data, tree, leaves)
    if model == Models.PULSE_BM:
        partial_args = (data, tree, leaves, distances)
    if model in {Models.PULSE, Models.PULSE_BM, Models.PULSE_SEPVARP, Models.PULSE_MU}:
        with Parallel(n_processes) as parallel:
            configurations = parallel(
                delayed(configuration_likelihood_unified)(
                    model, *partial_args, burst_locations, **kwargs
                )
                for burst_locations in burst_configs(n_nodes, max_n, min_bursts=min_n)
            )

    # Sort the list (minimal AIC_c first)
    configurations.sort(key=lambda x: x["aic_c"])

    # Metadata on the computations
    metadata = {
        "tree_filename": treefile,
        "data_filename": datafile,
        "with_meas_errors": scaled_meas_err,
        "min_n_pulses": min_n,
        "max_n_pulses": max_n,
        "data": list(data),
        "tree": tree.write(),
        "start_datetime": start_datetime.isoformat(),
        "finish_datetime": datetime.now().isoformat(),
        "n_processes": n_processes,
        "model": model,
        "mean_scaling_factor": rescale_factor,
        "var_scaling_factor": rescale_factor * rescale_factor,
        "lkl_scaling_factor": 1 / pow(rescale_factor, len(data)),
        "shift": shift
        # "_partial_args":partial_args,
        # "_kwargs":kwargs
    }
    if with_meas_errors:  # Include meas variance if available
        metadata["meas_vars"]: list(scaled_meas_err)

    # Best AICc configuration.
    min_aicc_configuration = configurations[0].copy()
    # Find a small enough step.
    proposed_eps = (
        min([np.abs(x) for x in min_aicc_configuration["fit_values"].values()]) / 5.0
    )
    # Avoids situation that given eps is too large (i.e. tiny var_p).
    eps = min(eps, proposed_eps)
    min_aicc_configuration["fit_variances_eps"] = numerical_variances(
        metadata, min_aicc_configuration, eps=eps
    )
    min_aicc_configuration["eps"] = eps

    # If var_p is computed separately, add it in.
    if model == Models.PULSE_SEPVARP:
        min_aicc_configuration["fit_values"]["var_p"] = min_aicc_configuration[
            "var_p_regime"
        ]
        min_aicc_configuration["fit_values"].move_to_end("var_d")
        min_aicc_configuration["fit_variances_eps"][
            "var_of_var_p"
        ] = min_aicc_configuration["var_var_p_regime"]
        min_aicc_configuration["fit_variances_eps"].move_to_end("var_of_var_d")

    return {
        "metadata": metadata,
        "configurations": configurations,
        "min_aicc_configuration": min_aicc_configuration,
    }


def _negloglkl_from_metadata(metadata, configuration):
    """Return neg log lkl function using metadata and configuration.
    
    Typical case: _negloglkl_from_metadata(res['metadata'], res['min_aicc_config']."""
    # Recover all needed parameters and structures
    model = metadata["model"]
    tree = Tree(metadata["tree"])
    data = metadata["data"]
    n_leaves, leaves = number_leaves(tree)
    n_nodes, nodes, distances = number_edges(tree)
    kwargs = {}
    if isinstance(metadata["with_meas_errors"], np.ndarray):
        kwargs["error_variances"] = metadata["with_meas_errors"]

    if model == Models.PULSE:
        return configuration_likelihood_neg_log_lkl_fun(
            model, data, tree, leaves, configuration["pulse_positions"], **kwargs
        )
    elif model == Models.PULSE_SEPVARP:
        return configuration_likelihood_neg_log_lkl_fun(
            model, data, tree, leaves, configuration["pulse_positions"], **kwargs
        )
    elif model == Models.PULSE_MU:
        return configuration_likelihood_neg_log_lkl_fun(
            model, data, tree, leaves, configuration["pulse_positions"], **kwargs
        )
    elif model == Models.PULSE_BM:
        return configuration_likelihood_neg_log_lkl_fun(
            model,
            data,
            tree,
            leaves,
            distances,
            configuration["pulse_positions"],
            **kwargs
        )
    elif model == Models.BM:
        return configuration_likelihood_neg_log_lkl_fun(
            model, data, tree, leaves, distances, **kwargs
        )


def numerical_variances(metadata, configuration, eps=1e-5):
    """Compute variances in parameters by inverting information matrix.
    
    We compute derivatives numerically; set eps to step size."""
    f = _negloglkl_from_metadata(metadata, configuration)
    crit_pt = list(configuration["fit_values"].values())
    hess = numerical_hessian(f, crit_pt)
    try:
        inv_hess = np.linalg.inv(hess)
    except LinAlgError:
        inv_hess = 1e15 * np.eye(len(crit_pt))
    variances = inv_hess.diagonal()
    labels = ["var_of_" + x for x in configuration["fit_values"].keys()]
    return OrderedDict(zip(labels, variances))


def group_by_regime(pulses, tree):
    """Returns nodes grouped by regime, give pulse positions.
    
    pulses = [(pulse_id, n_pulses), ..., ()]
    tree - phylogenetic tree
    nodes - list of all nodes in the tree."""
    accounted_for = []
    pulse_positions = [x[0] for x in pulses]
    # This corresponds to no pulses.
    pulse_positions.append(tree.id)
    equiv_classes = {}

    for pulse_position in pulse_positions:
        equiv_classes[pulse_position] = set()
        # for node in (nodes + [tree])[pulse_position].iter_descendants():
        for node in tree.iter_search_nodes(id=pulse_position):
            for leaf in node.iter_leaves():
                # if (node.is_leaf()):
                # print(node)
                equiv_classes[pulse_position].add(leaf)

    # Make the sets disjoint.
    for pos1 in pulse_positions:
        for pos2 in pulse_positions:
            if pos1 == pos2:
                continue
            else:
                if equiv_classes[pos1] <= equiv_classes[pos2]:
                    equiv_classes[pos2] = equiv_classes[pos2] - equiv_classes[pos1]

                if equiv_classes[pos2] <= equiv_classes[pos1]:
                    equiv_classes[pos1] = equiv_classes[pos1] - equiv_classes[pos2]

    # Get rid of empties!
    to_remove = []
    for equiv_class_id in equiv_classes:
        # check if empty
        if not equiv_classes[equiv_class_id]:
            to_remove.append(equiv_class_id)

    for equiv_class_id in to_remove:
        equiv_classes.pop(equiv_class_id)

    return equiv_classes


def group_ids(grouped_sets):
    """Returns list of lists, each inner list containing leaf IDs in regimes."""
    groups = []

    for group_key, group in grouped_sets.items():
        groups.append([])

        for node in group:
            groups[-1].append(node.leaf_id)

    return groups


def _pooled_var_p(grouped_ids, data):
    """Compute the pooled var_p, and the variance (error) of the estimated var_p."""
    grouped_vals = [[data[id_val] for id_val in group] for group in grouped_ids]

    numerator = 0.0
    denom = 0.0

    for group in grouped_vals:
        if len(group) > 1:
            numerator += (len(group) - 1) * np.var(group, ddof=1)
            denom += len(group) - 1

    estimated_variance = numerator / denom
    error_variance = 2.0 * estimated_variance ** 4 / denom

    return estimated_variance, error_variance


def pooled_var_p(pulses, tree, data):
    """Compute var_p and variance in estimate.
    
    pulses - e.g. [(12,1), (2, 1)]
    tree - phylogenetic tree with IDs and leaf IDs
    nodes - all nodes in tree
    data - data."""
    grouped_sets = group_by_regime(pulses, tree)
    grouped_ids = group_ids(grouped_sets)
    return _pooled_var_p(grouped_ids, data)


#### FUNCTION FOR RECOVERING ALL ANALYSES
def recover_analyses(pickletar_filename):
    with tarfile.open(pickletar_filename) as tar:
        analyses = pickle.load(tar.extractfile("list_of_pickled_files.p"))

        # Now replace filenames with actual data structures, and return that!
        for a in analyses.keys():
            if a == "tar_file":
                continue
            for b in analyses[a].keys():
                for c in analyses[a][b].keys():
                    analyses[a][b][c] = pickle.load(tar.extractfile(analyses[a][b][c]))

    return analyses
