import multiprocessing
import os

from joblib import delayed, Parallel

from Utilities.OSMnx_Utility import *
from Utilities.GTFS_Utility import *
from Utilities.General_Function import *
from Utilities.Data_Manipulation import *
import copy
import time
import tqdm


class InstanceNetwork:
    """
    The 'InstanceNetwork' class contains all information regarding the used intermodal transportation network, e.g.
    the network of Berlin. The functions of this class will combine the various mode networks and prepare the combined
    for the solution procedure.
    """

    # General attributes of this class
    place = ''
    connection_layer = 'connectionLayer'
    networks = []
    networksPath = 'data/networks/'
    baseMode = 'walk'
    start_time = time.time()

    # complete model of network in DiGraph format found in networkx
    G = nx.DiGraph()
    Graph_for_Preprocessing = nx.DiGraph()
    Graph_for_Preprocessing_Reverse = nx.DiGraph()

    # individual graph for each mode stored
    networkDict = {}
    setOfModes = []
    listOf_OSMNX_Modes = []

    # attributes of multimodal graph
    setOfNodes = []                 # format [(id, lat, lon),...]
    set_of_nodes_walk = set()
    setOfEdges = {}                 # format ( (from_node, to_node, mode) : [time, headway, distance]
    modifiedEdgesList_ArcFormulation = {}  # format {(node, node, mode) : [time, headway, distance, fixCost], ...}
    arcs_of_connectionLayer = []    # [(i,j,m)]
    modes_of_public_transport = []
    nodeSetID = []
    publicTransport = {}
    public_transport_modes = {}
    matchOfRouteIDandPublicTransportMode = {}   # format {routeID: public_transport_mode, ...} eg. {22453: 700, ...}
    networksAreInitialized = False

    # preprocessing
    graph_file_path = ""
    list_of_stored_punishes = []
    arcs_for_refined_search = []


    def __init__(self, place: str(), networks: []):
        self.place = place
        self.networks = networks
        printNewChaptToConsole('get input data')


    def initializeNetworks(self):
        """
        This function loads individual networks available from OpenStreetMap, except for the Public Transport Networks.
        :return: is safed directly to class attributes
        """

        # load all desired networks, specified in the attributes of this class
        for networkModes in self.networks:

            # update class attributes of contained networks
            self.setOfModes.append(networkModes)
            self.listOf_OSMNX_Modes.append(str(networkModes))

            # load network from memory if possible
            filepath = self.networksPath + self.place + ', ' + networkModes +'.graphML'
            if networkIsAccessibleAtDisk(filepath):
                self.networkDict[networkModes] = ox.load_graphml(filepath)

            # if network was not saved to memory before, load network from OpenStreetMap by the use of OSMnx 
            else:
                self.networkDict[networkModes] = downloadNetwork(self.place, networkModes, self.networksPath)

            # add travel time for each edge, according to mode
            # for walk, a average speed of  1.4 meters / second is assumed
            if networkModes == 'walk':
                G_temp = [] # ebunch
                for fromNode, toNode, attr in self.networkDict[networkModes].edges(data='length'):
                    G_temp.append((fromNode, toNode, {'travel_time': attr/(1.4*60), 'length': attr }))
                self.networkDict[networkModes].add_edges_from(G_temp)

            # for bike, an average speed of 4.2 meters / seconds is assumed
            elif networkModes =='bike':
                G_temp=[]
                for fromNode, toNode, attr in self.networkDict[networkModes].edges(data='length'):
                    G_temp.append((fromNode, toNode, {'travel_time': attr/(4.2*60), 'length': attr}))
                self.networkDict[networkModes].add_edges_from(G_temp)

            # the remaining mode is drive. As OSMnx also retrieves the speedlimits,
            # the inbuilt functions for calculating the travel time can be used
            else:
                self.networkDict[networkModes]= ox.add_edge_speeds(self.networkDict[networkModes], fallback=50, precision = 2)
                self.networkDict[networkModes] = ox.add_edge_travel_times(self.networkDict[networkModes], precision=2)

        # To fasten later steps, it is indicated that memory are already in memory
        self.networksAreInitialized = True

        print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to initialize networks')


    def mergePublicTransport(self):
        """
        This function adds the public transport networks (distinguished by modes, eg. Bus 52, Bus 53, ...) to the
        network instance.
        :return: return networks are directly assigned to class attributes
        """
        # get the networks of all modes of public transport
        publicTemporary = getPublicTransportation()

        # iterate through all mode categories (e.g. Bus, Tram, Subway, ...)
        for keys, mode_category in publicTemporary.items():
            self.publicTransport[str(keys)] = mode_category

        # get all edges and nodes of respective route network and add it to combined network
        # iterate through all mode cateogries: Tram, Subway, Bus, ...
        for mode_category, mode in self.publicTransport.items():

            # iterate through all modes: Bus 52, Bus 53
            for routeID, route in mode.items():

                # add attributes to class
                self.setOfModes.append(str(routeID))
                self.public_transport_modes [str(routeID)] = route

                # add nodes, which are stored in in the first element of the route list (format: ID, Lat, Lon)
                for node in route[0]:
                    self.setOfNodes.append([node[0], node[1], node[2]])

                # add edges, which are stored in the second element of the route list
                for edge, edge_attr in route[1].items():
                    edge_splitted = edge.split(":")
                    self.setOfEdges[(edge_splitted[0], edge_splitted[1], routeID)] = [
                        edge_attr['travelTime'], edge_attr['headway'], 0.0]


    def generateMultiModalGraph(self, parameters: {}):
        """
        This function takes the prepared input (the added networks retrieved from Public Transport and OpenStreetMap)
         and combines them into a Graph (as networkx DiGraph) and individual formats for edges and nodes for
        faster processing.
        :param parameters: Contains all features and attributes of the considered case, e.g. Berlin networks
        :return: nothing, as networks are directly assigned to class attributes
        """

        # if possible load complete generated Graph from memory
        edges_arc_formulation_file_path = self.networksPath +'setOfEdgesModified.json'
        if networkIsAccessibleAtDisk(edges_arc_formulation_file_path):
            self.modifiedEdgesList_ArcFormulation = self.get_dict_with_edgeKey(edges_arc_formulation_file_path)

        # if not in memory, create MulitModal Graph based on Class Attributes
        else:

            # if possible, load set of edges from memory
            filePathOfEdges = 'data/networks/setOfEdges.json'
            if networkIsAccessibleAtDisk(filePathOfEdges):
                self.getSavedSets()

            # generate set of all edges in multi modal network
            else:

                # retrieve all individual networks of OpenStreetMap
                self.initializeNetworks()

                # retrieve all individual networks of Public Transport network
                self.mergePublicTransport()

                # add each mode and the corresponding network to the mulitmodal graph
                for mode in self.setOfModes:

                    # special treatment for all osmnx data, as drive, walk and bike network share nodes already
                    if mode in self.listOf_OSMNX_Modes:

                        # get all nodes and edges of the viewed mode in desired format
                        setOfNodesByMode, setOfEdgesByMode = getListOfNodesOfGraph(self.networkDict[mode])

                        # add all nodes to MultiModal Graph
                        for nodes in setOfNodesByMode:
                            self.setOfNodes.append(nodes)

                        # add all edges to MultiModal Graph
                        if mode == 'drive':
                            for edge in setOfEdgesByMode:
                                travel_time = edge[2]['travel_time'] / 60  # convert into minutes
                                self.setOfEdges[(edge[0], edge[1], mode)] = [travel_time, 0.0, edge[2]['length']]
                        else:
                            self.setOfEdges[(edge[0], edge[1], mode)] = [edge[2]['travel_time'], 0.0, edge[2]['length']]

                    else:
                        connectingEdges = connectLayersPublicTransport(self.public_transport_modes[mode], mode,
                            self.networkDict[self.baseMode], self.baseMode)
                        print("Connection of mode " + mode + "-" + self.baseMode + " initialized")
                        for edge in connectingEdges.keys():
                            self.setOfEdges[(edge[0], edge[1], self.connection_layer)] = [connectingEdges[edge]['travel_time'],
                                                                                      0.0, connectingEdges[edge]['length']]
                self.setOfModes.append(self.connection_layer)

            self.saveMultiModalGraph()

            self.getModifiedNetwork(parameters)

        print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to get all json ready')

        self.initialize_base_network()

        self.initialize_graph_for_preprocessing(parameters)
        print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to get preprocessing graph ready')

        # get all modes of Public transport
        tempSet = set()
        for i, j, m in self.modifiedEdgesList_ArcFormulation.keys():
            if m not in self.listOf_OSMNX_Modes:
                tempSet.add(m)
            if m == self.connection_layer:
                self.arcs_of_connectionLayer.append((i, j, m))
        self.modes_of_public_transport=list(tempSet)
        print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to get input data in desired format')


    def initialize_base_network(self):
        if not self.networksAreInitialized:
            self.networkDict[self.baseMode] = ox.load_graphml(str(self.networksPath + 'Berlin, Germany, walk.graphML'))
            G_temp = []  # ebunch
            for fromNode, toNode, attr in self.networkDict[self.baseMode].edges(data='length'):
                G_temp.append((fromNode, toNode, {'travel_time': attr / (1.4 * 60)}))
            self.networkDict[self.baseMode].add_edges_from(G_temp)
            self.networksAreInitialized = True
            print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to initialize walk network')



    def add_Change_for_short_changes_drive_bike_public(self, parameters):
        filePathShortLinks = 'data/networks/short_changes_drive.json'
        shortLinks = {}
        start_time = time.time()
        if networkIsAccessibleAtDisk(filePathShortLinks):
            shortLinks = self.get_dict_with_edgeKey(filePathShortLinks)
        else:
            # get full network of walk
            G_change = nx.DiGraph()
            self.initialize_base_network()
            for i, j, travel_time in self.networkDict[self.baseMode].edges(data='travel_time'):
                G_change.add_edge(u_of_edge=str(i), v_of_edge=str(j), mode='walk', travel_time=travel_time)

            set_of_Nodes_PublicTransport = set()
            # add public transport nodes walk graph
            nodesToBeAdded = set()
            edgesToBeAdded = []
            waitingTimeOfMode = {}

            # add all nodes from other modes than walk
            for (i, j, m), attr in self.modifiedEdgesList_ArcFormulation.items():
                # for every indivdual network (so without walk and connection Layer
                if m != 'walk' and m!= self.connection_layer:
                    fromNode = i.split('+')[0]
                    toNode = j.split('+')[0]
                    if fromNode != toNode:
                        nodesToBeAdded.add(i)
                        nodesToBeAdded.add(j)
                # connect added nodes with walk network
                if m == self.connection_layer:
                    fromNode = i.split('+')[0]
                    fromMode = i.split('+')[1]
                    toNode = j.split('+')[0]
                    toMode = j.split('+')[1]
                    # arc from walk to public transport
                    if fromMode == 'walk':
                        edgesToBeAdded.append((fromNode, j, {'travel_time': attr[0], 'headway': attr[1], 'distance': attr[2], 'fix_cost' : attr[3]}))
                        waitingTimeOfMode[toMode] = attr[1]
                    # arc from public transport to walk
                    if toMode == 'walk':
                        edgesToBeAdded.append((i, toNode, {'travel_time': attr[0], 'headway': attr[1], 'distance': attr[2], 'fix_cost' : attr[3]}))

            waitingTimeOfMode['walk'] = 0
            waitingTimeOfMode['bike'] = parameters['waiting_time_bike']
            waitingTimeOfMode['drive'] = parameters['waiting_time_drive']
            G_change.add_nodes_from(nodesToBeAdded)
            # add connectionLayer
            G_change.add_edges_from(edgesToBeAdded)
            # calculate shortest for all
            allShortestPath = dict(nx.all_pairs_dijkstra_path_length(G_change, weight='travel_time',
                                                                     cutoff=3))  # everthing more than 5 min is not really close)
            # only retain the links for connecting the modes
            shortLinks = {}
            for source in allShortestPath.keys():
                source_split = source.split('+')
                if len(source_split) != 1:
                    for target in allShortestPath[source].keys():
                        target_split=target.split('+')
                        if len(target_split) != 1:
                            if source_split[1] != target_split[1]:
                                shortLinks[source, target, self.connection_layer] = {
                                    'travel_time': allShortestPath[source][target],
                                    'headway': waitingTimeOfMode[target.split('+')[1]]}
            self.safe_dict_with_edgeKey(safeDict=shortLinks, dataFilePath=filePathShortLinks)
        print("--- %s seconds ---" % round((time.time() - self.start_time), 2) + ' to short changes ready')
        # filter PT
        for edge, attr in shortLinks.items():
            fix_cost = 0
            if edge[1].split('+')[1] == 'drive':
                fix_cost = parameters['fixCost_Taxi']
            self.modifiedEdgesList_ArcFormulation[edge] = [attr['travel_time'], attr['headway'], 0, fix_cost]

    def node_part_of_PublicTransport(self, node):
        node_is_part = True
        node_splitted = node.split('+')
        if len(node_splitted) == 1:
            node_is_part = False
        return node_is_part


    def safe_dict_with_edgeKey(self, safeDict: {}, dataFilePath):
        newKey = str()
        newDict = {}
        for edge, edge_attr in safeDict.items():
            newKey = ""
            for i, var in enumerate(edge):
                newKey += var
                if i != len(edge) - 1:
                    newKey += '&'
            newDict[newKey] = edge_attr
        with open(dataFilePath, 'w') as file:
            json.dump(newDict, file)

    def get_dict_with_edgeKey(self, filePath):
        returnDict = {}
        with open(filePath, 'r') as file:
            tempDict = json.load(file)
        for edge, edge_attr in tempDict.items():
            # edge back to original format
            edge_split = tuple(edge.split('&'))
            # add
            returnDict[edge_split] = edge_attr
        return returnDict

    def saveMultiModalGraph(self):
        dataFileNodes = open('data/networks/setOfNodes.json', "w")
        json.dump(self.setOfNodes, dataFileNodes)
        dataFileNodes.close()
        dataFileModes = open('data/networks/setOfModes.json', "w")
        json.dump(self.setOfModes, dataFileModes)
        dataFileModes.close()
        dataFileEdges = open('data/networks/setOfEdges.json', "w")
        dictToBeSaved = {}
        for edge in self.setOfEdges.keys():
            dictToBeSaved[str(edge[0]) + "-" + str(edge[1]) + "-" + str(edge[2])] = self.setOfEdges[edge]
        json.dump(dictToBeSaved, dataFileEdges)
        dataFileEdges.close()

    def safeSetOfChanges(self):
        dataFileChanges = open('data/networks/setOfChanges.json', "w")
        json.dump(self.listOfChangeConstraints, dataFileChanges)
        dataFileChanges.close()

    def getNetwork(self, mode: str):
        return self.networkDict[mode]

    def getSetOfNodes(self):
        return self.setOfNodes

    def getSavedSets(self):
        dataFileNodes = open('data/networks/setOfNodes.json', 'r')
        tempList = json.load(dataFileNodes)
        for j in tempList:
            self.setOfNodes.append((j[0], j[1], j[2]))
        self.setOfNodes = list(set(self.setOfNodes))
        dataFileNodes.close()
        dataFileModes = open('data/networks/setOfModes.json', 'r')
        self.setOfModes = json.load(dataFileModes)
        dataFileModes.close()
        dataFileEdges = open('data/networks/setOfEdges.json', 'r')
        setOfEdgesTemp = json.load(dataFileEdges)
        for edge in setOfEdgesTemp.keys():
            edgeSplitted = edge.split("-")
            self.setOfEdges[(edgeSplitted[0], edgeSplitted[1], edgeSplitted[2])] = setOfEdgesTemp[edge]
        dataFileEdges.close()

    def getSavedChanges(self):
        dataFileChanges = open('data/networks/setOfChanges.json', 'r')
        tempList = json.load(dataFileChanges)
        dataFileChanges.close()
        for i in tempList:
            self.setOfModeChanges.append((i[0], i[1], i[2]))
            self.listOfChangeConstraints.append(i)

    def initializePublicTransport(self):
        return getPublicTransportation()

    def getModifiedNetwork(self, parameters):
        fixCost_tx = parameters['fixCost_Taxi']
        modifiedEdgeList = {}
        modesSortedByNodes = {}
        # prepare edge formulation
        for edge in self.setOfEdges.keys():
            if edge[2] != self.connection_layer:
                fromNode = str(edge[0] + '+' + edge[2])
                toNode = str(edge[1] + '+' + edge[2])
                mode = str(edge[2])
                modifiedEdgeList[(fromNode, toNode, mode)] = self.setOfEdges[edge]
                if edge[0] in modesSortedByNodes.keys():
                    modesSortedByNodes[edge[0]].append(edge[2])
                else:
                    modesSortedByNodes[edge[0]] = [edge[2]]
                if edge[1] in modesSortedByNodes.keys():
                    modesSortedByNodes[edge[1]].append(edge[2])
                else:
                    modesSortedByNodes[edge[1]] = [edge[2]]

            if edge[2] == 'drive':
                modifiedEdgeList[(fromNode, toNode, mode)].append(fixCost_tx)
            else:
                modifiedEdgeList[(fromNode, toNode, mode)].append(0)

        setOfEdgesOrdered = {}
        for (i, j, m) in self.setOfEdges.keys():
            # from_arc
            if j in setOfEdgesOrdered.keys():
                setOfEdgesOrdered[j]['arcEnteringNode'].append((i, j, m))
            else:
                setOfEdgesOrdered[j] = {'arcEnteringNode': [(i, j, m)], 'arcLeavingNode': []}
            # to_arc
            if i in setOfEdgesOrdered.keys():
                setOfEdgesOrdered[i]['arcLeavingNode'].append((i, j, m))
            else:
                setOfEdgesOrdered[i] = {'arcEnteringNode': [], 'arcLeavingNode': [(i, j, m)]}
        fromNode = str()
        toNode = str()
        travel_time = float()
        headway = float()
        travel_distance = float()
        fixCost = float()
        for node in setOfEdgesOrdered.keys():
            for (i, j, m) in setOfEdgesOrdered[node]['arcEnteringNode']:
                for (a, b, n) in setOfEdgesOrdered[node]['arcLeavingNode']:
                    fixCost = 0.0

                    # entering edge is from other network
                    if m == self.connection_layer:
                        if n != 'walk' and n != self.connection_layer:
                            fromNode = str(i) + '+' + 'walk'
                            toNode = str(j) + '+' + str(n)
                            travel_time = self.setOfEdges[(i, j, m)][0]
                            if n == 'bike':
                                headway = parameters['waiting_time_bike']
                            else:
                                headway = self.setOfEdges[(a, b, n)][1]
                            travel_distance = self.setOfEdges[(i, j, m)][2]
                            if n == 'drive':
                                fixCost = fixCost_tx
                                headway = parameters['waiting_time_drive']
                            modifiedEdgeList[(fromNode, toNode, self.connection_layer)] = [travel_time, headway,
                                                                                       travel_distance, fixCost]

                    # leaving edge is going back to walk network
                    elif n == self.connection_layer:
                        if m != 'walk' and m != self.connection_layer:
                            fromNode = str(j) + '+' + str(m)
                            toNode = str(b) + '+' + 'walk'
                            travel_time = self.setOfEdges[(i, j, m)][0]
                            headway = 0
                            travel_distance = self.setOfEdges[(i, j, m)][2]
                            fixCost = 0
                            modifiedEdgeList[(fromNode, toNode, self.connection_layer)] = [travel_time, headway,
                                                                                           travel_distance, fixCost]

                    else:
                        # for changes within PT
                        if m != n:
                            fromNode = str(j) + '+' + str(m)
                            toNode = str(j) + '+' + str(n)
                            travel_time = 0
                            headway = self.setOfEdges[(a, b, n)][1]
                            travel_distance = 0
                            if n == 'drive':
                                fixCost = fixCost_tx
                                headway = parameters['waiting_time_drive']
                            if n == 'bike':
                                headway = parameters['waiting_time_bike']
                            modifiedEdgeList[(fromNode, toNode, self.connection_layer)] = [travel_time, headway,
                                                                                       travel_distance, fixCost]
        self.modifiedEdgesList_ArcFormulation = modifiedEdgeList
        self.add_Change_for_short_changes_drive_bike_public(parameters)
        dictToBeSafed = {}
        safeKey = str()
        for i, j, m in modifiedEdgeList:
            safeKey = str(i) + '&' + str(j) + '&' + str(m)
            dictToBeSafed[safeKey] = modifiedEdgeList[i, j, m]
        dataFileEdgesModified = open('data/networks/setOfEdgesModified.json', "w")
        json.dump(dictToBeSafed, dataFileEdgesModified)
        dataFileEdgesModified.close()

    def initialize_graph_for_preprocessing(self, parameters):
        beta_as_string = str(parameters['beta'])
        dir_path = 'data/preprocessing'
        file_path = dir_path + '/graph_of_' + beta_as_string + '_and_' + str(
            parameters['maxNumber_of_Changes']) + '_max_changes'
        self.graph_file_path = file_path

        if not networkIsAccessibleAtDisk(file_path):
            # Graph to be returned
            G = nx.DiGraph(directed=True)
            modesConsideredWalking = ['walk', self.connection_layer]

            # initialize variables before to reduce memory
            disutility = float()
            toMode = str()
            extraDisutilityFixCost = float()
            if parameters['maxNumber_of_Changes'] >= 10:
                maxNumber_of_Changes = 10
            else:
                maxNumber_of_Changes = parameters['maxNumber_of_Changes']

            # add all edges of combined network to graph with calculated edge disutility
            for (fromNode, toNode, mode) in self.modifiedEdgesList_ArcFormulation.keys():
                disutility = 0
                extraDisutilityFixCost = 0

                # disutility due to in-vehicle time
                if mode not in modesConsideredWalking and mode != 'bike':
                    disutility += parameters['beta'][0] * self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][
                        0]  # in-vehicle

                # disutility due to walking (for connectionLayer and walking)
                elif mode != 'bike':
                    disutility += parameters['beta'][1] * self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][
                        0]  # walking time

                # disutility due to waiting and cost entering a mode (only considered in connectionLayer)
                if mode == self.connection_layer:
                    disutility += parameters['beta'][2] * 0.5 * self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][1]  # waiting time
                    disutility += parameters['beta'][3] * self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][3]  # fix cost for drive as arc dependent

                    # cost for entering public transport and bike are not arc dependent, but a fraction can be assigned to approximate the real disutility
                    toMode = toNode.split('+')[1]

                    if toMode not in modesConsideredWalking and toMode != 'bike' and toMode != 'drive':
                        extraDisutilityFixCost = (1 / maxNumber_of_Changes) * parameters[
                            'fixCost_PublicTransport'] * parameters['beta'][3]
                    if toMode == 'bike':
                        extraDisutilityFixCost = ((1 / maxNumber_of_Changes) * parameters['fixCost_Bike'] *
                                                 parameters['beta'][3])

                # variable cost for using taxi
                if mode == 'drive':
                    disutility += (parameters['beta'][3] * parameters['varCost_Taxi'] *
                                  self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][
                                      2])

                # disutility for using bike
                if mode == 'bike':
                    disutility += parameters['beta'][4] * self.modifiedEdgesList_ArcFormulation[(fromNode, toNode, mode)][0]

                # add arc with calculated disutility to the DiGraph
                punish_base = round((disutility + extraDisutilityFixCost), 5)
                disutility = round(disutility, 5)

                if mode == self.connection_layer:
                    G.add_edge(fromNode, toNode, weight=punish_base, arc_disutility=disutility,
                               weightWithChangePunishment=punish_base, mode=mode)
                else:
                    G.add_edge(fromNode, toNode, weight=(disutility + extraDisutilityFixCost), arc_disutility=disutility,
                               weightWithChangePunishment=(disutility + extraDisutilityFixCost), mode=mode)

            # assign generated Graph to class attributes
            self.Graph_for_Preprocessing = G
            self.Graph_for_Preprocessing_Reverse = self.Graph_for_Preprocessing.reverse()
            if not os.path.isfile(file_path):
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                nx.write_gpickle(G, file_path)

        else:
            self.Graph_for_Preprocessing = nx.read_gpickle(file_path)
            self.Graph_for_Preprocessing_Reverse = self.Graph_for_Preprocessing.reverse()

        self.list_of_stored_punishes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60,  80, 100]
        self.Graph_for_Preprocessing = add_prestored_punishments(G=self.Graph_for_Preprocessing,
                                                                 list_of_punishment=self.list_of_stored_punishes)
        self.Graph_for_Preprocessing_Reverse = add_prestored_punishments(G=self.Graph_for_Preprocessing_Reverse,
                                                                 list_of_punishment=self.list_of_stored_punishes)

        self.Graph_for_Preprocessing = add_arc_attribute_for_change_test(self.Graph_for_Preprocessing)
        for (i, j, m) in self.arcs_of_connectionLayer:
            # all changes except for changes from and to Public Transport
            if i.split('+')[1] in ['bike', 'drive', 'walk'] and j.split('+')[1] in ['bike', 'drive', 'walk']:
                self.arcs_for_refined_search.append((i,j,m))

    def get_attr_of_node(self, name):
        # need initialized walking network
        self.initialize_base_network()
        print(nx.get_node_attributes(self.networkDict[self.baseMode], name))




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[x:x+n] for x in range(0, len(lst), n)]


def add_prestored_punishments(G, list_of_punishment):
    for (i, j) in G.edges():
        punish_base = G[i][j]['weight']
        if G[i][j]['mode'] == 'connectionLayer':
            for change_punish in list_of_punishment:
                punish_stored = 'punish_' + str(change_punish)
                G[i][j][punish_stored] = punish_base + change_punish
        else:
            for change_punish in list_of_punishment:
                punish_stored = 'punish_' + str(change_punish)
                G[i][j][punish_stored] = punish_base
        G[i][j]['refined_search_punishment'] = punish_base
        G[i][j]['cur_punishment'] = 0
    return G

def add_arc_attribute_for_change_test(G):
    for (i, j) in G.edges():
        punish_base = G[i][j]['weight']
        if G[i][j]['mode'] == 'connectionLayer':
            G[i][j]['check_number_of_changes'] = 1
        else:
            G[i][j]['check_number_of_changes'] = 0
    return G

