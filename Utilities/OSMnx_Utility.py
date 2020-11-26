import osmnx as ox
import networkx as nx
from Utilities.Solution_Procedure import *
import gurobipy as gb


def downloadNetwork(place: str, modeOfTransport: str, networksPath: str) -> bool:
    ox.config(all_oneway=True, log_console=True, use_cache=True)
    pathForStorage = networksPath + place + ', ' + modeOfTransport
    G = ox.graph_from_place(place, network_type=modeOfTransport, simplify=True)
    ox.save_graphml(G, filepath=pathForStorage)
    return G


def networkIsAccessibleAtDisk(path: str) -> bool:
    try:
        f = open(path, mode='r')
        f.close()
    except FileNotFoundError:
        return False
    except IOError:
        return False
    return True


def getListOfNodesOfGraph(Graph):
    """
    By calling this function, all nodes and edges of a specific Graph can be retreived. It is ensured,
    that there are no more than one connection between each pair of nodes. Transfrom into usable lists with only one
    connection, its kept track of many arcs are removed - see console. Issue of using a multiDiGraph where multiple
    connections between Nodes are possible and thus are removed
    :param Graph: networkx DiGraph
    :return: all nodes and edges of a graph, respectively as list
    """

    # Get Edges and Nodes from Graph
    setOfEdges = Graph.edges(data=True)
    setOfNodes = Graph.nodes()
    listOfNodes = []
    setOfLinks = []
    setOfCosts = {}
    numberOfEdgesRemoved = 0

    # Determine the fastest connection between each pair of nodes
    for edge in setOfEdges:
        linkToBeAdded = (edge[0], edge[1])
        if linkToBeAdded in setOfCosts.keys():
            numberOfEdgesRemoved += 1
            if setOfCosts[linkToBeAdded]['length'] > edge[2]['length']:
                setOfCosts[linkToBeAdded]['length'] = edge[2]['length']
        else:
            setOfCosts[linkToBeAdded] = edge[2]

    # only safe the determined fastest connection, in order to avoid multiple connections
    for edges, attr in setOfCosts.items():
        setOfLinks.append((edges[0], edges[1], attr))

    print(str(numberOfEdgesRemoved) + ' Edges were removed!')

    # get nodes into desired format of (ID, latitutde, longitude)
    for node in setOfNodes._nodes.keys():
        listOfNodes.append((str(node), setOfNodes._nodes[node]['y'], setOfNodes._nodes[node]['x']))

    return listOfNodes, setOfLinks
