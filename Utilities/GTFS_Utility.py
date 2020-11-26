from Utilities.OSMnx_Utility import *

import sys
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import geopandas as gp
import matplotlib
from datetime import time, timedelta, datetime

import gtfs_kit as gk


def getHeadway(routeType):
    # U Bahn
    if routeType == 400:
        return 5
    # S Bahn
    elif routeType == 109:
        return 5
    # Tram
    elif routeType == 900:
        return 7
    # Bus
    else:
        return 10


def storeGTFSasJson(routeType):
    path = Path(r'Data/networks/GTFS/gtfs.zip')
    file = 'gtfs.zip'

    feed = gk.read_feed(path, dist_units='km')

    if (gk.valid_date('20200706')) is True:
        date = '20200706'

    feed = gk.drop_zombies(feed)

    # getting all necessary dataframes,
    routeDf = feed.get_routes(date)
    routeDF = routeDf.filter(items=['route_id', 'route_type'])
    # active routes on monday of the given mode of transportation
    routeDF = routeDF[routeDF['route_type'] == routeType]
    tripsDf = feed.get_trips(date)
    tripsDf = tripsDf.filter(items=['route_id', 'service_id', 'trip_id'])
    stop_timesDf = feed.get_stop_times(date)
    stop_timesDf = stop_timesDf.filter(items=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])
    stopsDf = feed.get_stops(date)
    stopsDf = stopsDf.filter(items=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'])

    # storing routes in list
    routes = []
    for index, row in routeDF.iterrows():
        routes.append(row['route_id'])

    # creating the result Dict
    publicTransport = {}
    for i in range(0, len(routes)):
        publicTransport[routes[i]] = []

    # iterating through all the routes; creating nodes and edges
    print('routes of route_type ' + str(routeType) + ' are getting initialized...')
    for route in publicTransport:

        # get all trips for the route
        tripsOfRouteDf = tripsDf[tripsDf['route_id'] == route]

        # creating list of trip
        trips = []
        for index, row in tripsOfRouteDf.iterrows():
            trips.append(row['trip_id'])
        tripsOfRouteDf = tripsOfRouteDf[tripsOfRouteDf.trip_id.isin(trips)]

        # getting stop_times of the route
        stop_timesOfRouteDf = stop_timesDf[stop_timesDf.trip_id.isin(trips)]

        # getting edges
        # dictionary for edges: Dict of Dict
        edges = {}
        # find longest trip to get all edges

        maxLength = 0
        maxTrip_id = ''
        for trip in trips:
            stops_inTrips = stop_timesOfRouteDf[stop_timesOfRouteDf['trip_id'] == trip]
            length = len(stops_inTrips)
            if length > maxLength:
                maxLength = length
                maxTrip_id = trip

        # filter stop times by trip with the most stops
        stop_timesOfTrip = stop_timesOfRouteDf[stop_timesOfRouteDf['trip_id'] == maxTrip_id]

        # ordering by stop sequence
        stop_timesOfTrip = stop_timesOfTrip.filter(items=['stop_id', 'stop_sequence', 'arrival_time'])
        stop_timesOfTrip = stop_timesOfTrip.sort_values(by=['stop_sequence'])

        # creating list of tuples to iterate through the stops
        listOfStop_times = list(map(tuple, stop_timesOfTrip.to_numpy()))

        valueEdges = {}
        # enter values of edges in dictionary: cost, travel time, headway
        for i in range(0, len(listOfStop_times) - 1):
            edges[(listOfStop_times[i][0], listOfStop_times[i + 1][0])] = valueEdges
            edges.get((listOfStop_times[i][0], listOfStop_times[i + 1][0]))['cost'] = 0
            try:
                timeDestination = datetime.strptime(listOfStop_times[i + 1][2], '%H:%M:%S')
                timeStart = datetime.strptime(listOfStop_times[i][2], '%H:%M:%S')
                travelTimedelta = timeDestination - timeStart
                travelTime = travelTimedelta.total_seconds() / 60
            except ValueError:
                # hours can have values > 24: convert these values
                hourDestination = int(listOfStop_times[i + 1][2][0] + listOfStop_times[i + 1][2][1])
                hourStart = int(listOfStop_times[i][2][0] + listOfStop_times[i][2][1])
                minuteDestination = int(listOfStop_times[i + 1][2][3] + listOfStop_times[i + 1][2][4])
                minuteStart = int(listOfStop_times[i][2][3] + listOfStop_times[i][2][4])
                secDestination = int(listOfStop_times[i + 1][2][6] + listOfStop_times[i + 1][2][7])
                secStart = int(listOfStop_times[i][2][6] + listOfStop_times[i][2][7])
                travelTime = (hourDestination * 60 + minuteDestination + secDestination / 60) - (
                        hourStart * 60 + minuteStart + secStart / 60)
            if travelTime == 0:
                travelTime = 0.1
            edges.get((listOfStop_times[i][0], listOfStop_times[i + 1][0]))['travelTime'] = travelTime

            headway = getHeadway(routeType)
            edges.get((listOfStop_times[i][0], listOfStop_times[i + 1][0]))['headway'] = headway

        # get nodes from edges
        # iterate through edges and store ids in a set
        setNodeids = set()
        for nodetuple in edges:
            setNodeids.add(nodetuple[0])
            setNodeids.add(nodetuple[1])

        stops = stopsDf[stopsDf.stop_id.isin(setNodeids)].filter(items=['stop_id', 'stop_lat', 'stop_lon'])
        listOfNodes = list(map(tuple, stops.to_numpy()))
        publicTransport.get(route).append(listOfNodes)

        # create edges in both directions
        edgesComplete = {}
        for key in edges:
            edgesComplete[key] = edges[key]
            edgesComplete[(key[1], key[0])] = edges[key]

        # key in edges needs to be a string in order to store it as json
        edgesString = {}
        for key in edgesComplete:
            edgesString[key[0] + ":" + key[1]] = edgesComplete[key]

        publicTransport.get(route).append(edgesString)

    print('number of active routes: ' + str(len(publicTransport)))

    dataFile = open("data/networks/" + str(routeType) + "gtfs.json", "w")
    json.dump(publicTransport, dataFile)
    dataFile = open('data/networks/' + str(routeType) + "gtfs.json", 'r')
    print("route" + str(routeType) + "initialized")
    return dataFile.read()


def getJson(routeType):
    dataFile = open('data/networks/' + str(routeType) + 'gtfs.json', 'r')
    return json.load(dataFile)


def getPublicTransportation():
    publicTransportationDict = {}
    transportationModes = [400, 109, 700, 900]
    for mode in transportationModes:
        if networkIsAccessibleAtDisk('data/networks/' + str(mode) + 'gtfs.json'):
            publicTransportationDict[mode] = getJson(mode)
        else:
            publicTransportationDict[mode] = storeGTFSasJson(mode)
    return publicTransportationDict


def networkIsAccessibleAtDisk(path: str) -> bool:
    try:
        f = open(path, mode='r')
        f.close()
    except FileNotFoundError:
        return False
    except IOError:
        return False
    return True


def getPolygon():
    publicTransportDict = getPublicTransportation()
    listOfStops = []
    for route in publicTransportDict:
        for node in publicTransportDict[route]:
            listOfStops.append(node[0])

    path = Path(r'Data/networks/GTFS/gtfs.zip')
    file = 'gtfs.zip'

    feed = gk.read_feed(path, dist_units='km')
    return gk.compute_convex_hull(feed, stop_ids=listOfStops)
