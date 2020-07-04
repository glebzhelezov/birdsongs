""" BOAT: Birds On A Tree module

This module allows you to compute the likelihood of phenotype means on the
leaves of an evolutionary tree, given a burst-on-edges model. A better name
would be "Bursts On A Tree."

Please direct questions and comments to Gleb Zhelezov and Jay McEntee.

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
            |     \ What a birdsong!
            |      \       /
            |       \     / ____O
            |        \     .' ./
            |   _.,-~"\  .',/~'
            <-~"   _.,-~" ~ |
^"~-,._.,-~"^"~-,._\       /,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,.
~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-
^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,.
~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._.,-~"^"~-,._

(ASCII art found online.)

"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
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
        neglkl += (0.5*np.log(2*np.pi*(var_p + errors[leaf_id]))
            + (data[leaf_id] - mu_hat)**2 / (2*(var_p + errors[leaf_id])))
    
    return neglkl


def neglkl_marginal_var_p(var_p, pulse_positions, tree, data, errors):
    """Return negloglikelihood of var_p marginal--contribs from all regimes.
    """
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
    
    return [
    [scaling*scaling*res[0][0], scaling*scaling*res[0][1], rescaled_mean],
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
                    equiv_classes[pos2] = (equiv_classes[pos2] -
                        equiv_classes[pos1])
                if equiv_classes[pos2] <= equiv_classes[pos1]:
                    equiv_classes[pos1] = (equiv_classes[pos1] - 
                        equiv_classes[pos2])

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
