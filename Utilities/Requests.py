import csv
import json
from pathlib import Path
import sys
import random
import geopy.distance
import numpy as np
import matplotlib.pyplot as plt
import Utilities.General_Function
import pandas as pd
import networkx as nx
import osmnx as ox

from Utilities.Data_Manipulation import networkIsAccessibleAtDisk


def getRequests():
    path_CSV = 'data/requests/BerlinSimRequests.csv'  # change the name
    requestData = list()
    with open(path_CSV, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) > 0:
                requestData.append((row[0], row[1]))
    requestData.pop(0)
    return requestData


def getGeoCoordinates():
    geoCoordinatesDict = dict()
    path_CSV = 'data/requests/BerlinMATSim.csv'  # change the name
    requestData = list()
    with open(path_CSV, newline='') as csvfile:
        fileCSV = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in fileCSV:
            if len(row) > 0:
                geoCoordinatesDict[row[0]] = (row[1], row[2])
    del geoCoordinatesDict['id']
    return geoCoordinatesDict


def getRequest(BerlinInstance):
    listOfRequests = getRequests()
    for (fromNode, toNode) in listOfRequests:
        itsInThere = False
        for node in BerlinInstance.setOfNodes:
            if fromNode == node[0] or toNode == node[0]:
                itsInThere = True
        if itsInThere:
            print('No worries')
        else:
            print('You are screwed')
    return listOfRequests


def initializeRequests():
    requests = dict()
    listOfRequests = getRequests()
    dictOfGeos = getGeoCoordinates()
    for (fromNode, toNode) in listOfRequests:
        requests[(fromNode, toNode)] = {'fromLat': float(dictOfGeos[fromNode][0]),
                                        'fromLon': float(dictOfGeos[fromNode][1]),
                                        'toLat': float(dictOfGeos[toNode][0]), 'toLon': float(dictOfGeos[toNode][1])}
    return requests


def getNumberRequests(BerlinInstance, number):
    pathFile = "data/requests/" + str(number) + "requests.json"
    if not networkIsAccessibleAtDisk(pathFile):
        requests = initializeRequests()
        numberRequests = 0
        requestsnumber = []
        while numberRequests < number:
            # random choice of a request
            selected = random.choice(list(requests.keys()))
            # check if request origin and destination point is max 200m away from a node in the network
            if ox.get_nearest_node(BerlinInstance.networkDict[BerlinInstance.baseMode],
                                   (requests.get(selected)['fromLat'], requests.get(selected)['fromLon']),
                                   return_dist=True, method="haversine")[1] < 200:
                if ox.get_nearest_node(BerlinInstance.networkDict[BerlinInstance.baseMode],
                                       (requests.get(selected)['toLat'], requests.get(selected)['toLon']),
                                       return_dist=True, method="haversine")[1] < 200:
                    # check if distance between o and d is bigger than 500m
                    if ox.distance.great_circle_vec(requests.get(selected)['fromLat'], requests.get(selected)['fromLon'],
                                           requests.get(selected)['toLat'], requests.get(selected)['toLon']) > 500:
                        # if the request is feasible it is added to the list
                        try:
                            nx.dijkstra_path(BerlinInstance.networkDict[BerlinInstance.baseMode],
                                             ox.get_nearest_node(BerlinInstance.networkDict[BerlinInstance.baseMode],
                                                                 (requests.get(selected)['fromLat'],
                                                                  requests.get(selected)['fromLon'])),
                                             ox.get_nearest_node(BerlinInstance.networkDict[BerlinInstance.baseMode],
                                                                 (requests.get(selected)['toLat'],
                                                                  requests.get(selected)['toLon'])))
                            request = {}
                            request['fromLat'] = requests.get(selected)['fromLat']
                            request['fromLon'] = requests.get(selected)['fromLon']
                            request['toLat'] = requests.get(selected)['toLat']
                            request['toLon'] = requests.get(selected)['toLon']
                            requestsnumber.append(request)
                            numberRequests = numberRequests + 1
                            print(numberRequests)
                        except nx.NetworkXNoPath:
                            print("this request is infeasible")
        dataFile = open(pathFile, "w")
        json.dump(requestsnumber, dataFile)
        evaluateNumberRequests(requestsnumber)
    dataFile = open(pathFile, "r")
    print('requests initialized')
    return json.load(dataFile)


def evaluateNumberRequests(requests):
    distances = []
    for pair in requests:
        origin = (pair['fromLat'], pair['fromLon'])
        destination = (pair['toLat'], pair['toLon'])
        distances.append(geopy.distance.distance(origin, destination).km)
        print()
    x = np.linspace(0, len(requests), len(requests))
    plt.plot(x, distances, "o")
    plt.show()


def getBetaSet():
    df = pd.read_csv(r'data/beta_set.csv')
    betaList = []
    for index, rows in df.iterrows():
        betas = [rows.Beta_1, rows.Beta_2, rows.Beta_3, rows.Beta_4, rows.Beta_5]
        betaList.append(betas)
    return betaList
