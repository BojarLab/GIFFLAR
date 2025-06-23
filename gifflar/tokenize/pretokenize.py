import sys
import io

import networkx as nx
from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream
from glycowork.motif.graph import glycan_to_nxGraph

from glyles.iupac.IUPACLexer import IUPACLexer
from glyles.iupac.IUPACParser import IUPACParser
from glyles.glycans.poly.anltr_error_listener import GlyLESErrorListener


def graph_to_token_stream_int(graph):
    """converts glycan graph back to IUPAC-condensed format\n
    | Arguments:
    | :-
    | graph (networkx object): glycan graph\n
    | Returns:
    | :-
    | Returns glycan in IUPAC-condensed format (string)
    """

    def assign_depth(G: nx.DiGraph, idx: int):
        """Assigns depth to each node in the graph recursively. Stored as the node attribute "depth".\n
        | Arguments:
        | :-
        | G (networkx object): glycan graph
        | idx (int): node index\n
        | Returns:
        | :-
        | Returns depth of the node
        """
        min_depth = float("inf")
        children = list(G.neighbors(idx))
        if len(children) == 0:  # if it's a leaf node, the depth is zero
            G.nodes[idx]["depth"] = 0
            return 0
        for child in children:  # if it's not a leaf node, the depth is the minimum depth of its children + 1
            min_depth = min(assign_depth(G, child) + 1, min_depth)
        G.nodes[idx]["depth"] = min_depth
        return min_depth

    def dfs_to_string(G: nx.DiGraph, idx: int):
        """Converts the DFT tree of a glycan graph to IUPAC-condensed format recursively.\n
        | Arguments:
        | :-
        | G (networkx object): DFS tree of a glycan graph
        | idx (int): node index\n
        | Returns:
        | :-
        | Returns glycan in IUPAC-condensed format (string)
        """
        output = [graph.nodes[idx]["string_labels"]]
        # put braces around the string describing linkages
        if (output[0][0] in "?abn" or output[0][0].isdigit()) and (output[0][-1] == "?" or output[0][-1].isdigit()):
            output = ["("] + output + [")"]
        # sort kids from shallow to deep to have deepest as main branch in the end
        children = list(sorted(G.neighbors(idx), key=lambda x: G.nodes[x]["depth"]))
        if len(children) == 0:
            return output

        for child in children[:-1]:  # iterate over all children except the last one and put them in branching brackets
            output = ["["] + dfs_to_string(G, child) + ["]"] + output
        # put the last child in front of the output, without brackets
        output = dfs_to_string(G, children[-1]) + output
        return output

    # get the root node index, assuming the root node has the highest index
    root_idx = max(list(graph.nodes.keys()))

    # get the DFS tree of the graph
    dfs = nx.dfs_tree(graph, root_idx)

    # assign depth to each node
    assign_depth(dfs, root_idx)

    # convert the DFS tree to IUPAC-condensed format
    return dfs_to_string(dfs, root_idx)


def graph_to_token_stream(graph):
    """converts glycan graph back to IUPAC-condensed format\n

    Assumptions:
    1. The root node is the one with the highest index.

    | Arguments:
    | :-
    | graph (networkx object): glycan graph\n
    | Returns:
    | :-
    | Returns glycan in IUPAC-condensed format (string)
    """
    if nx.number_connected_components(graph) > 1:
        parts = [graph.subgraph(sorted(c)) for c in nx.connected_components(graph)]
        len_org = len(parts[-1])
        for p in range(len(parts) - 1):
            H = nx.Graph()
            H.add_nodes_from(sorted(parts[p].nodes(data=True)))
            H.add_edges_from(parts[p].edges(data=True))
            parts[p] = nx.relabel_nodes(H, {pn: pn - len_org for pn in H.nodes()})
            len_org += len(H)
        tokens = []
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                tokens += graph_to_token_stream_int(p)
            else:
                tokens += ["{"] + graph_to_token_stream_int(p) + ["}"]
        return tokens
    else:
        return graph_to_token_stream_int(graph)


class PreTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __call__(self, input_: str):
        raise NotImplementedError()


class GlycoworkPreTokenizer(PreTokenizer):
    def __call__(self, iupac: str):
        return graph_to_token_stream(glycan_to_nxGraph(iupac))


class GrammarPreTokenizer(PreTokenizer):
    def __call__(self, iupac: str):
        iupac = iupac.strip().replace(" ", "")
        token = CommonTokenStream(lexer := IUPACLexer(InputStream(data="#" + iupac + "#")))
        parser = IUPACParser(token)

        lexer.removeErrorListeners()
        lexer.addErrorListener(GlyLESErrorListener())
        parser.removeErrorListeners()
        parser.addErrorListener(GlyLESErrorListener())

        try:
            parser.start()
        except Exception as e:
            return None

        return [t.text for t in token.tokens[1:-2]]


class CharacterPreTokenizer(PreTokenizer):
    def __call__(self, iupac: str):
        return list(iupac)
