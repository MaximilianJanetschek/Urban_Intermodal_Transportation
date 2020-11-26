import json
import networkx as nx


def checkForEquivalence(tour1, tour2):
    if tour1 == tour2:
        print("Gurobi route and NX route are equal")
    else:
        print("Gurobi route and NX route are different")


def processInput(totalSet, index, setOfEdges):
    subListOfChanges = list()
    for (from_node, i, from_mode) in totalSet[index]:
        for (j, to_node, to_mode) in setOfEdges.keys():
            if j == i:
                if from_mode != to_mode:
                    subListOfChanges.append((j, from_mode, to_mode, from_node, i, from_mode, j, to_node, to_mode))
    return subListOfChanges


def getSavedDict(filepath):
    dataFileOfDict = open(filepath, 'r')
    tempDict = json.load(dataFileOfDict)
    dataFileOfDict.close()
    return tempDict


def safeDict(filepath: str, dictToBeSaved: dict()):
    dataFileOfDict = open(filepath, "w")
    json.dump(dictToBeSaved, dataFileOfDict)
    dataFileOfDict.close()


def networkIsAccessibleAtDisk(path: str) -> bool:
    try:
        f = open(path, mode='r')
        f.close()
    except FileNotFoundError:
        return False
    except IOError:
        return False
    return True


def printNewChaptToConsole(chapterName: str()):
    print()
    print('#########################################################')
    print(chapterName)
    print('#########################################################')
    print()


def transform_TourNodes_in_TourArcs(tour_in_nodes: list()) -> list():
    tour_in_arcs = list()
    for i in range(len(tour_in_nodes)):
        if len(tour_in_arcs) == 0:
            tour_in_arcs.append((tour_in_nodes[0], tour_in_nodes[1]))
        elif i == 1:
            continue
        else:
            fromNode, toNode = tour_in_arcs[-1]
            tour_in_arcs.append((toNode, tour_in_nodes[i]))
    return tour_in_arcs

def single_source_multi_target_paths (G, sublist, weight):
    return_dict = {}
    for source in sublist:
        return_dict[source] = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    return return_dict

def multiprocessing_single_source_shortest_paths(G, G_reverse,  source_origin, source_destination, cutoff, weight, key, listOfReachableNodes):
    if key == 'origin':
        listOfReachableNodes['origin'] = nx.single_source_dijkstra_path_length(G=G, source=source_origin, cutoff=cutoff, weight=weight)
    else:
        listOfReachableNodes['destination'] = nx.single_source_dijkstra_path_length(G=G_reverse, source=source_destination,
                cutoff= cutoff, weight=weight)
    return 'done'
