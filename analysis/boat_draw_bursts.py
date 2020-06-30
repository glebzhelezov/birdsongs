import boat_submission as boat
from ete3 import Tree, TreeStyle, TextFace, NodeStyle, CircleFace, FaceContainer
import numpy as np

# With precomputed likelihood of each configutation, make graph
def plot_burst_tree(nwkfile, datafile, lkls, extra_str="", title=None):
    
    if title is None:
        title = datafile
    
    # Copy data and tree
    tree_copy, data_copy = boat.load_treefile_datafile(nwkfile, datafile)
    data_copy = np.array(data_copy)
    n_leaves_copy, leaves_copy = boat.number_leaves(tree_copy)  # Numbers leaves
    n_nodes_copy, nodes_copy, distances_copy = boat.number_edges(
        tree_copy
    )  # Numbers edges

    # Sort likelihoods
    lkls_sorted = sorted(lkls, key=lambda x: x[4])
    branch_confidence = np.zeros(n_nodes_copy)
    normalization = 0.0
    for config in sorted(lkls, key=lambda l: l[0], reverse=False):
        info_score = config[4]
        config_weight = np.exp(-info_score / 2)
        normalization += config_weight
        # computed_weighted_lkl = max(config[0], 0.)
        # normalization = normalization + computed_weighted_lkl

        for burst_location in config[2]:
            # branch_confidence[burst_location[0]] += computed_weighted_lkl
            branch_confidence[burst_location[0]] += config_weight

    branch_confidence = branch_confidence / normalization

    ts = TreeStyle()
    # ts.title.add_face(TextFace("Pulse support | Max {} bursts\nData: {} (prelim.)".format(max_n_bursts, datafile), fsize=10), column=0)
    ts.title.add_face(
        TextFace(
            "Pulse support\nData: {}".format(title), fsize=10
        ),
        column=0,
    )
    ts.show_leaf_name = True
    ts.legend.add_face(CircleFace(5, "blue", style="sphere"), column=0)
    ts.legend.add_face(
        TextFace(" One pulse under min AICc model", fsize=7), column=1
    )
    ts.legend.add_face(TextFace("w.xyz", fgcolor="blue", fsize=7), column=0)
    ts.legend.add_face(TextFace(" Pulse support", fsize=7), column=1)
    ts.legend_position = 1
    ts.complete_branch_lines_when_necessary = False

    for n in tree_copy.iter_descendants():
        nstyle = NodeStyle()
        # nstyle["fgcolor"] = "red"
        # nstyle["size"] = max(2, 5*branch_confidence[n.id])
        # n.img_style["size"]
        nstyle["hz_line_width"] = 10 * branch_confidence[n.id]
        nstyle["size"] = 0
        label = "{:.3f}".format(branch_confidence[n.id])
        n.add_face(
            TextFace(label, fgcolor="blue", fsize=7),
            column=0,
            position="branch-top",
        )
        n.add_face(
            TextFace(n.id, fgcolor="red", fsize=5),
            column=0,
            position="branch-bottom",
        )
        n.set_style(nstyle)

    # Label the active edges for the overall lowest AICc score
    for burst in lkls_sorted[0][2]:
        print(burst)
        pos = burst[0]
        nbursts = burst[1]
        nodes_copy[pos].img_style["shape"] = "sphere"
        nodes_copy[pos].img_style["size"] = 10 * nbursts
        nodes_copy[pos].img_style["fgcolor"] = "blue"

    tree_copy.render(
        datafile[:-4] + "_" + nwkfile[:-4] + "_graph" + extra_str + ".png",
        w=800,
        units="px",
        tree_style=ts,
    )


def ascii_edge_labels(nwkfile):
    """Return an ASCII tree with edge labels and leaf names."""
    tree_copy = Tree(nwkfile)
    boat.number_leaves(tree_copy)  # Numbers leaves
    boat.number_edges(tree_copy)  # Numbers edges
    return tree_copy.get_ascii(attributes=["name", "id"])


def pulse_supports(lkls, n_nodes):
    """Return pulse support on each edge.

    Return pulse support on each edge, given precomputed likelihoods for
    each configuration.
    """

    branch_confidence = np.zeros(n_nodes)
    normalization = 0.0
    for config in sorted(lkls, key=lambda l: l[0], reverse=False):
        info_score = config[4]
        config_weight = np.exp(-info_score / 2)
        normalization += config_weight
        # computed_weighted_lkl = max(config[0], 0.)
        # normalization = normalization + computed_weighted_lkl

        for burst_location in config[2]:
            # branch_confidence[burst_location[0]] += computed_weighted_lkl
            branch_confidence[burst_location[0]] += config_weight

    branch_confidence = branch_confidence / normalization

    return branch_confidence


def likeliest_configuration(lkls):
    """Return likeliest pulse configuration."""
    lkls_sorted = sorted(lkls, key=lambda x: x[4])
    return lkls_sorted[0][2].copy()


def pulse_info_string(nwkfile, lkls, show_tree=True, show_details=True):
    """Return edge labels, confidences, most probable configuration."""

    # Copy data and tree
    tree = Tree(nwkfile)
    boat.number_leaves(tree)  # Numbers leaves
    n_nodes = boat.number_edges(tree)[0]  # Numbers edges + get number of nodes

    string_to_return = ""
    # Print out the tree
    if show_tree:
        string_to_return += "Edge labels:\n"
        string_to_return += ascii_edge_labels(nwkfile)
        string_to_return += "\n\n"

    if show_details:
        # Label the active edges for the overall lowest AICc score
        string_to_return += "Likeliest pulse combination:\n"

        for burst in likeliest_configuration(lkls):
            string_to_return += "\tEdge #{} has {} pulses.\n".format(
                burst[0], burst[1]
            )

        # Compute confidences on each edge, then output it.
        string_to_return += "\nPulse confidence on edge:\n"
        branch_confidence = pulse_supports(lkls, n_nodes)

        for i in range(n_nodes):
            string_to_return += "\tEdge #{}:\t{}".format(
                i, branch_confidence[i]
            )
            # Mark confidence >0.1
            if branch_confidence[i] > 0.1:
                string_to_return += "\t> 0.1"
            string_to_return += "\n"

    return string_to_return
