from Utilities.Data_Retrieval import *
from typing import Tuple
from math import sqrt
from math import radians, sin, cos, acos
import osmnx as ox
from Utilities.OSMnx_Utility import *
import json
from pathlib import Path


def connectLayers(fromNetwork, fromMode, toNetwork, toMode):
    filePath = ""
    filePath += str(fromMode) + "-" + str(toMode) + ".json"
    connectionLayer = {}
    print('data/networks/connectionLayers' + filePath)
    print(networkIsAccessibleAtDisk('data/networks/connectionLayers/' + filePath))
    if networkIsAccessibleAtDisk('data/networks/connectionLayers/' + filePath):
        connectionLayer = getSavedLayer(filePath)
    else:
        connectionLayer = calculateConnectionLayer(fromNetwork, toNetwork, filePath)
    return connectionLayer


def calculateConnectionLayer(fromNetwork, toNetwork, file):
    # save it as a file
    connectionLayer = {}
    distance = float('inf')
    numberOfNodes = len(fromNetwork.nodes())
    for nodeFrom in fromNetwork.nodes():
        if nodeFrom not in toNetwork.nodes():
            fromNode = fromNetwork.nodes._nodes[nodeFrom]
            nearestNode, distance = ox.get_nearest_node(toNetwork, point=(fromNode['y'], fromNode['x']),
                                                        return_dist=True, method='haversine')
            travel_time = distance / (1.4 * 60)  # convert distance in min
            if (distance != 0):
                connectionLayer[nodeFrom, nearestNode] = {'travel_time': travel_time, 'length': distance}
                connectionLayer[nearestNode, nodeFrom] = {'travel_time': travel_time, 'length': distance}
            print(str(numberOfNodes) + ' with distance ' + str(distance))
            numberOfNodes -= 1
        else:
            numberOfNodes -= 1
    saveLayerAsJson(file, connectionLayer)
    return connectionLayer


def connectLayersPublicTransport(fromRoute, fromMode, toNetwork, toMode):
    file = fromMode + "-" + toMode + ".json"
    connectionLayer = {}
    if networkIsAccessibleAtDisk('data/networks/connectionLayers/' + file):
        connectionLayer = getSavedLayer(file)
    else:
        connectionLayer = calculateConnectionLayerPublicTransport(fromRoute, toNetwork, file)
    return connectionLayer


def calculateConnectionLayerPublicTransport(fromRoute, toNetwork, file):
    # save it as a file
    connectionLayer = {}
    distance = float('inf')
    numberOfNodes = len(fromRoute[0])
    for fromNode in fromRoute[0]:
        nearestNode, distance = ox.get_nearest_node(toNetwork, point=(fromNode[1], fromNode[2]), return_dist=True,
                                                    method='haversine')
        travel_time = distance / (1.4 * 60)  # convert distance in min
        if (distance != 0):
            connectionLayer[fromNode[0], nearestNode] = {'travel_time': travel_time, 'length': distance}
            connectionLayer[nearestNode, fromNode[0]] = {'travel_time': travel_time, 'length': distance}
        if (distance >= 100):
            print(str(numberOfNodes) + ' with distance ' + str(distance) + 'lat: ' + str(fromNode[1]) + 'lon: ' + str(
                fromNode[2]))
        numberOfNodes -= 1
    saveLayerAsJson(file, connectionLayer)
    return connectionLayer


def getSavedLayer(filePath):
    dataFile = open('data/networks/connectionLayers/' + filePath, 'r')
    dictData = json.load(dataFile)
    returnDict = dict()
    dataFile.close()
    for edge in dictData.keys():
        edgeSplitted = edge.split("-")
        returnDict[(edgeSplitted[0], edgeSplitted[1])] = dictData[edge]
    return returnDict


def saveLayerAsJson(filePath, connectionLayer):
    layerToBeSafed = {}
    keyForSaving = ""
    for fromNode, toNode in connectionLayer.keys():
        keyForSaving = str(fromNode) + "-" + str(toNode)
        layerToBeSafed[keyForSaving] = connectionLayer[(fromNode, toNode)]
    dataFile = open('data/networks/connectionLayers/' + filePath, "w")
    json.dump(layerToBeSafed, dataFile)
    dataFile = open('data/networks/connectionLayers/' + filePath, 'r')
    dataFile.close()


def euclidean_distance(origin, target):
    slat = radians(float(origin['y']))
    slon = radians(float(origin['x']))
    elat = radians(float(target['y']))
    elon = radians(float(target['x']))

    dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
    return dist


def networkIsAccessibleAtDisk(path: str) -> bool:
    try:
        f = open(path, mode='r')
        f.close()
    except FileNotFoundError:
        return False
    except IOError:
        return False
    return True

