import threading

import gurobipy as gb
from sklearn.feature_extraction import DictVectorizer

from Utilities.Data_Retrieval import *
import csv
from Utilities.General_Function import *
from Utilities.GTFS_Utility import *
import osmnx as ox
import networkx as nx
from networkx import NetworkXNoPath
from numpy import array
from Utilities.Drawing import *
import time
import os


def multi_mode_optimization_in_Arc_fromulation(origin, destination, instanceNetwork, parameters: dict()) -> list():
    """
    Given the input, this function will calculate the optimal path between the origin and destination in.
    :param origin: geos of starting point
    :param destination: geos of end point
    :param instanceNetwork: multimodal network
    :param parameters: instance specific parameters such as objective function parameters
    :return: tour from origin to destination as list
    """

    # get nearest nodes to origin in Graph
    destination, distance_d = ox.get_nearest_node(instanceNetwork.networkDict[instanceNetwork.baseMode], destination,
                                                  return_dist=True, method='haversine')
    distance_from_network_to_destination = distance_d / (1.4 * 60)  # convert distance in tavel_time

    # get nearest node to destination in Graph
    origin, distance_o = ox.get_nearest_node(instanceNetwork.networkDict[instanceNetwork.baseMode], origin,
                                             return_dist=True, method='haversine')
    distance_from_origin_to_network = distance_o / (1.4 * 60)  # convert distance in travel_time

    # get optimized solution with given parameters and weights
    dictOfSolution = get_optimized_solution(networkInstance=instanceNetwork, parameters=parameters, origin=origin,
                                                destination=destination,
                                                maxNumberOfChanges=parameters['maxNumber_of_Changes'])

    # create folder if necessary to save solution
    if not os.path.exists('results'):
        os.mkdir('results')

    # write solution to csv
    filePath_CSV = 'results/optimalPathInformation-' + str(parameters['beta']) + '.csv'
    write_Solution_to_CSV(file_path=filePath_CSV, beta=parameters['beta'], dictOfSolution=dictOfSolution, origin=origin,
                          destination=destination)

    return [dictOfSolution['orderedSolutionTour'], dictOfSolution['w']]


def get_optimized_solution(networkInstance, parameters: dict(), origin, destination, maxNumberOfChanges):
    """

    :param networkInstance:
    :param parameters:
    :param origin:
    :param destination:
    :param maxNumberOfChanges:
    :return:
    """

    printNewChaptToConsole('start solution procedure') # quick update to console

    dictOfSolution = dict()

    # get id of origin and destination nodes into desired format (original id + mode_type)
    origin = str(origin) + '+walk'
    destination = str(destination) + '+walk'

    # short cut retrival edges
    set_of_Edges = networkInstance.modifiedEdgesList_ArcFormulation
    modesConsideredWaking_bike = ['walk', 'connectionLayer', 'bike']

    # prepare sets necessary for model build up
    modesConsideredWaking = ['walk', 'connectionLayer']
    modesNotConsideredPT = ['walk', 'connectionLayer', 'bike', 'drive']

    # as the berlin intermodal network is quite big, a preprocessing algorithm is applied to reduce the graph size
    reachableNodes, feasibleSolution, preprocessing_time, heuristic_objvalue = preprocessInput(setOfEdges=set_of_Edges,
                                                       modesConsideredWalking=modesConsideredWaking,
                                                       modesConsideredNotPT=modesNotConsideredPT, parameters=parameters,
                                                       origin=origin, destination=destination,
                                                       instanceNetwork=networkInstance)

    # only remain those edges in the graph, which nodes where not removed from grpah by the preprocessing (reachable nodes)
    modifiedEdgeListUpdated = dict()
    start_time = time.time()
    for i, j, m in set_of_Edges.keys():
        if i in reachableNodes and i != destination:
            if j in reachableNodes and j != origin:
                modifiedEdgeListUpdated[i, j, m] = networkInstance.modifiedEdgesList_ArcFormulation[i, j, m]

    # calculate the set for each constraint and remove all nodes which are only connected to one other node (see additional preprocessing)
    dict_of_constraints = ['shortest_path']
    previous_length = len(modifiedEdgeListUpdated.keys())
    shortest_path_constraint = remove_unnecessary_edges(modifiedEdgeListUpdated, origin, destination)
    new_length = len(modifiedEdgeListUpdated.keys())
    print('Shortened Edges by ' + str(round(((previous_length-new_length)/previous_length)*100, 4)) + ' % in '
          + str(round(time.time()-start_time, 2)) + ' seconds')

    # get set for the edge decision variables ready
    linksNode = gb.tuplelist()
    for links in modifiedEdgeListUpdated.keys():
        linksNode.append(links)

    # get all sets for the set up ready
    modesConsideredPT = set(m for (i, j, m) in linksNode.select('*', '*', '*') if m not in modesNotConsideredPT)
    setOfNonWalkingModes = set(m for (i, j, m) in linksNode.select('*', '*', '*') if m not in modesConsideredWaking)
    setOfNonWalkingModes_bike = set(m for (i, j, m) in linksNode.select('*', '*', '*') if m not in modesConsideredWaking_bike)

    # Set up Gurobi
    multiModalModel = gb.Model('Multi Modal Optimization')
    # multiModalModel.params.outputflag = 0

    # binary variable y_ijm
    y = multiModalModel.addVars(linksNode, vtype=gb.GRB.BINARY, name="y")

    # binary variable PB
    PT = multiModalModel.addVar(vtype=gb.GRB.BINARY, name="PT")

    # binary variable BK
    BK = multiModalModel.addVar(vtype=gb.GRB.BINARY, name="BK")

    # safe attributes of the preprocessing and network instance in the model, for later use in callbacks
    multiModalModel._vars = y
    multiModalModel._graphParameters = parameters
    multiModalModel._graphOrigin = origin
    multiModalModel._graphDestination = destination
    multiModalModel._graphInstance = networkInstance
    multiModalModel._preprocessingSolution = feasibleSolution
    multiModalModel._FeasibleNodes = reachableNodes
    multiModalModel._use_better_solution = False
    multiModalModel._linksNode = linksNode
    multiModalModel._time = time.time()

    # set heuristic start solution value
    PT.start = feasibleSolution['PublicTransport']
    BK.start = feasibleSolution['Bike']
    for i, j, m in linksNode.select('*', '*', '*'):
        if (i, j) in feasibleSolution['Tour']:
            y[i, j, m].start = 1
        else:
            y[i, j, m].start = 0

    # Objective Function
    obj = gb.LinExpr()
    # in-vehicle  time
    obj += parameters['beta'][0] * (gb.quicksum(set_of_Edges[(i, j, m)][0] * y[i, j, m] for (i, j, m) in
                                                linksNode.select('*', '*',
                                                                 setOfNonWalkingModes_bike)))
    # walking time
    obj += parameters['beta'][1] * (gb.quicksum(set_of_Edges[(i, j, m)][0] * y[i, j, m] for (i, j, m) in
                                                linksNode.select('*', '*', modesConsideredWaking)))
    # waiting time
    obj += parameters['beta'][2] * (0.5 * gb.quicksum(set_of_Edges[(i, j, m)][1] * y[i, j, m] for (i, j, m) in
                                                      linksNode.select('*', '*', 'connectionLayer')))
    # costs
    obj += parameters['beta'][3] * (parameters['fixCost_PublicTransport'] * PT + parameters['fixCost_Bike'] * BK
                                    + gb.quicksum(
                parameters['varCost_Taxi'] * set_of_Edges[i, j, m][2] * y[i, j, m] for (i, j, m) in
                linksNode.select('*', '*', 'drive'))
                                    + gb.quicksum(y[i, j, m] * set_of_Edges[(i, j, m)][3] for (i, j, m) in
                                                  linksNode.select('*', '*', 'connectionLayer')))
    # biking time
    obj += parameters['beta'][4] * (
        gb.quicksum(set_of_Edges[(i, j, m)][0] * y[i, j, m] for (i, j, m) in linksNode.select('*', '*', 'bike')))

    # adjust model Parameters
    multiModalModel.setObjective(obj, gb.GRB.MINIMIZE)
    multiModalModel.Params.MIPFocus = 3
    multiModalModel.Params.Presolve = 1
    multiModalModel.Params.Cuts = 1
    multiModalModel.Params.Method = 2
    multiModalModel.update()

    # calculate the set for each constraint
    dict_of_constraints = get_dict_of_constraints(modifiedEdgeListUpdated=modifiedEdgeListUpdated,
                                                  modesConsideredPublicTransport=modesConsideredPT,
                                                  shortest_path_constraints=shortest_path_constraint)


    # Shortest Path Constraints
    for node in dict_of_constraints['shortest_path'].keys():
        multiModalModel.addConstr(
            gb.quicksum(y[a, b, c] for a, b, c in dict_of_constraints['shortest_path'][node]['arcEnteringNode'])
            - gb.quicksum(y[d, e, f] for d, e, f in dict_of_constraints['shortest_path'][node]['arcLeavingNode']) ==
            (-1 if node == str(origin) else 1 if node == str(destination) else 0),
            f'shortest path constriant for {node} ')

        # only allow one edge to enter any node (only needed for nodes, where more than two arcs enter respective node)
        if len(dict_of_constraints['shortest_path'][node]['arcEnteringNode'])>= 2:
            multiModalModel.addConstr(
                gb.quicksum(y[i, j, m] for i, j, m in dict_of_constraints['shortest_path'][node]['arcEnteringNode']) <= 1,
                'performanceConstraint')

    # limit number of transits
    multiModalModel.addConstr(
        gb.quicksum(y[i, j, m] for i, j, m in linksNode.select('*', '*', 'connectionLayer')) <= maxNumberOfChanges,
        'limit number of changes')

    # Bike transport binary variable calculation
    # determining variable by entering arcs
    multiModalModel.addConstr(
        gb.quicksum(y[i, j, m] for i, j, m in dict_of_constraints['bike_indicator']['entering'].keys()) <= (
                maxNumberOfChanges - 1) * BK, 'bikeConstraint Entering')

    # determining variable by leaving arcs (added to allow for more cuts)
    multiModalModel.addConstr(
        gb.quicksum(y[i, j, m] for i, j, m in dict_of_constraints['bike_indicator']['leaving'].keys()) <= (
                maxNumberOfChanges - 1) * BK, 'bikeConstraint Leaving')

    # Public transport binary variable calculation
    # determining variable by entering arcs
    multiModalModel.addConstr(gb.quicksum(
        y[i, j, m] for (i, j, m) in dict_of_constraints['publicTransport_indicator']['entering'].keys()) <= (
                                      maxNumberOfChanges - 1) * PT, 'publicTransportConstraint Entering')
    # determining variable by leaving arcs (added to allow for more cuts)
    multiModalModel.addConstr(
        gb.quicksum(y[i, j, m] for (i, j, m) in dict_of_constraints['publicTransport_indicator']['leaving'].keys()) <= (
                maxNumberOfChanges - 1) * PT, 'publicTransportConstraint Leaving')


    print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' gurobi model is build up')

    multiModalModel.update()

    multiModalModel.optimize(callback_for_cuts)

    # if the preprocessing was off, i.e. more than 10% deviation from best solution determined by gurobi, and if the optimization takes more than 20 seconds the graph is reduced again.
    # the logic is the same as in the preprocessing
    while multiModalModel._use_better_solution:
        print('Rerun with tighter formulation')
        # reduce the y variables
        linksNode = update_gurobi_model(multiModalModel, linksNode)

        # update the available y variables
        multiModalModel._linksNode = linksNode
        multiModalModel.update()

        # refine search
        multiModalModel._use_better_solution = False
        multiModalModel.Params.Cuts = 0

        # optimize again
        multiModalModel.optimize(callback_for_cuts)

    print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' optimized with gurobi')
    optimization_time = round((time.time()-start_time),2)

    # apply post-processing (only for documentation)
    dictOfSolution = post_processing(multiModalModel, linksNode, PT, BK, origin, destination, parameters, optimization_time, setOfNonWalkingModes, modesConsideredWaking, set_of_Edges,
                    preprocessing_time,  heuristic_objvalue, y)


    return dictOfSolution


def orderTour(unorderedSolution: list(), destination, origin) -> list():
    """
    Function to order the result gained from gurobi into a meaning full path. Input variables, as they are named.
    :return: ordered tour of nodes
    """
    orderedResultList = list()
    temp_list = []
    tourNotClosed = True
    counter = 0
    stoping_criterion = len(unorderedSolution)*2
    while tourNotClosed:
        if len(temp_list) == 0:
            for node in unorderedSolution:
                if str(node[0]) == str(origin):
                    temp_list.append(node)
                    unorderedSolution.remove(node)
        elif temp_list[-1][1] == str(destination):
            tourNotClosed = False
        else:
            for node in unorderedSolution:
                if node[0] == temp_list[-1][1]:
                    temp_list.append(node)
                    unorderedSolution.remove(node)
        if counter >= stoping_criterion:
            print(origin)
            print(unorderedSolution)
            print(temp_list)
            print(destination)
            raise SyntaxError
        counter += 1

    for (i,j,m) in temp_list:
        orderedResultList.append((i.split('+')[0], j.split('+')[0], m))

    return orderedResultList


def post_processing(multiModalModel, linksNode, PT, BK, origin, destination, parameters, optimization_time,
                    setOfNonWalkingModes, modesConsideredWaking, set_of_Edges,
                    preprocessing_time,  heuristic_objvalue, y):
    """
    Function that serves the documentation of all results into a csv-File.
    """

    dictOfSolution = dict()
    if multiModalModel.status == gb.GRB.Status.OPTIMAL:
        dictOfSolution['y'] = list()
        dictOfSolution['w'] = list()
        for i, j, m in linksNode:
            if y[i, j, m].x >= 0.8:
                dictOfSolution['y'].append((i, j, m))
                if m == 'connectionLayer':
                    dictOfSolution['w'].append((i.split('+')[0], i.split('+')[1], j.split('+')[1]))
        dictOfSolution['pt'] = PT.x
        dictOfSolution['bk'] = BK.x

        dictOfSolution['orderedSolutionTour'] = orderTour(dictOfSolution['y'], destination=destination,
                                      origin=origin)

        printNewChaptToConsole('end solution procedure')

        # get all objective function criteria values by recalculating their objective function contribution
        inVehicleTime = 0
        inVehicleTimePT = 0
        inVehicleTimeDrive = 0
        inVehicleTimeBK = 0
        for i, j, m in linksNode.select('*', '*', setOfNonWalkingModes):
            if m == 'drive':
                inVehicleTimeDrive += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
                inVehicleTime += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
            elif m == 'bike':
                inVehicleTimeBK += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
                inVehicleTime += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
            else:
                inVehicleTimePT += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
                inVehicleTime += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
        dictOfSolution['Disutility'] = multiModalModel.objVal
        dictOfSolution['inVehicleTime'] = inVehicleTime
        dictOfSolution['inVehicleTimePT'] = inVehicleTimePT
        dictOfSolution['inVehicleTimeDrive'] = inVehicleTimeDrive
        dictOfSolution['inVehicleTimeBK'] = inVehicleTimeBK
        walkingTime = 0
        for (i, j, m) in linksNode.select('*', '*', modesConsideredWaking):
            walkingTime += set_of_Edges[(i, j, m)][0] * y[i, j, m].x
        dictOfSolution['walkingTime'] = walkingTime
        waitingTime = 0
        for (i, j, m) in linksNode.select('*', '*', 'connectionLayer'):
            waitingTime += 0.5 * set_of_Edges[(i, j, m)][1] * y[i, j, m].x
        dictOfSolution['waitingTime'] = waitingTime
        cost = parameters['fixCost_PublicTransport'] * PT.x + parameters['fixCost_Bike'] * BK.x
        for (i, j, m) in linksNode.select('*', '*', 'drive'):
            cost += parameters['varCost_Taxi'] * set_of_Edges[i, j, m][2] * y[i, j, m].x
        for (i, j, m) in linksNode.select('*', '*', 'connectionLayer'):
            cost += y[i, j, m].x * set_of_Edges[(i, j, m)][3]
        dictOfSolution['cost'] = cost

        dictOfSolution['optimizationTime'] = optimization_time
        dictOfSolution['preprocessingTime'] = preprocessing_time
        dictOfSolution['Gap'] = round(((heuristic_objvalue - multiModalModel.objval)/multiModalModel.objval),3)
        dictOfSolution['maxNumber_of_Changes'] = parameters['maxNumber_of_Changes']

        return dictOfSolution


def preprocessInput(setOfEdges, modesConsideredWalking, modesConsideredNotPT, parameters: dict(), origin, destination,
                    instanceNetwork):
    """
    This functions calculates an incumbent, feasible solution. It's based on a shortest path formulation of the problem
    and checks it's feasibility afterwards. If the found solution is infeasible, the punishment for changes between
    different modes (e.g. bus 42, tram 16 and taxi) is increased, to reduce changes. The punishment is increased until
    a feasible solution is obtained. Afterwards it is tried to further improve the solution.
    """
    start_time = time.time()

    tourIsNotFeasible = True
    changePunisher = 0
    dictFeasibleSolution = dict()
    dict_Solution_Parameters = dict()

    # Get prepared Graph in networkx DiGraph Graph
    G = instanceNetwork.Graph_for_Preprocessing

    print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' for graph')

    added_artificial_cost = False
    counter = 0

    # indicator to determine if the punishment on change arcs needs to be reset
    first_increase_punisher = True


    while tourIsNotFeasible:
        counter += 1

        # adapt G to punishment
        if changePunisher != 0:
            print('change punishment ' + str(changePunisher))

            first_increase_punisher = False

            # determine whether there is a saved graph with at most 10 or 2.5 deviation of the determined punishement
            stored_punish = ''
            for i in instanceNetwork.list_of_stored_punishes:
                # due to the lower additional punishment on change, the effect of deviation between determined and
                # and saved punishment should be small
                if i <= 50:
                    if abs(changePunisher-i) <= 2.5:
                        stored_punish = 'punish_' + str(i)
                        break

                # if the punishement is higher, we can allow for a higher deviation.
                else:
                    if abs(changePunisher-i) <= 10:
                        stored_punish = 'punish_' + str(i)
                        break

            if stored_punish != '':
                    print('Use pre-stored punish of ' + stored_punish)
                    # get shortest Path from origin to destination
                    length, shortestPath = nx.bidirectional_dijkstra(G, origin, destination, weight=stored_punish)

            else:
                increase_cost_on_change_arcs(G, changePunisher, instanceNetwork.arcs_of_connectionLayer, first_increase_punisher)

                # get shortest Path from origin to destination
                length, shortestPath = nx.bidirectional_dijkstra(G, origin, destination, weight='weightWithChangePunishment')

        else:
            # get shortest Path from origin to destination
            length, shortestPath = nx.bidirectional_dijkstra(G, origin, destination, weight='weight')

        # transform shortest Path into list of edges
        shortestPathTour = transform_TourNodes_in_TourArcs(shortestPath)

        # manually check the numbers for the relaxed variables
        dict_Solution_Parameters = get_solution_parameters(G, shortestPathTour, modesConsideredNotPT,
                                                           parameters=parameters)

        if dict_Solution_Parameters['Number_of_Changes'] <= parameters['maxNumber_of_Changes']:
            tourIsNotFeasible = False
            dictFeasibleSolution = dict_Solution_Parameters

        # as solution is not feasible, the punishment needs to be increased
        else:
            if changePunisher <= 0.01:
                changePunisher = (dict_Solution_Parameters['Number_of_Changes'] - parameters[
                    'maxNumber_of_Changes']) * (dict_Solution_Parameters['Path_Disutility'] / 20)
            elif changePunisher >= (dict_Solution_Parameters['Path_Disutility'] / 2):
                changePunisher = changePunisher * 1.5
            else:
                changePunisher = changePunisher * 2
            added_artificial_cost = True

        print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' to calculate ' + str(
            counter) + ' heurisitc solution')

    # determine reduced Graph based on found solution
    setOfFeasibleNodes = nodes_of_reduced_Graph(G, origin, destination, dictFeasibleSolution, instanceNetwork, parameters)
    print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' finished preprocessing')

    preprocessing_time = round((time.time()-start_time),2)

    return setOfFeasibleNodes, dictFeasibleSolution, preprocessing_time, dict_Solution_Parameters['Path_Disutility']


def nodes_of_reduced_Graph(G, origin, destination, dict_Solution_Parameters, networkInstance, parameters) -> set():
    """
    Function that uses the determined feasible solution to remvoe any node, of which the approximated limit is higher than
    the identified solution. See preprocessing part 2.
    """
    removed_edges_G = []
    test_time = time.time()

    # remove edges entering origin and leaving destination (-> cannot be part of tour)
    for edge in G.in_edges(origin, data=True):
        removed_edges_G.append(edge)
    for edge in G.out_edges(destination, data=True):
        removed_edges_G.append(edge)
    G.remove_edges_from(removed_edges_G)

    # same just reverse for the reversed Graph
    G_reinvert = networkInstance.Graph_for_Preprocessing_Reverse # graph is used for one-source multiple destinations.

    # remove edges entering origin and leaving destination (-> cannot be part of tour), but reverse logic
    removed_edges_G_reverse = []
    for edge in G_reinvert.out_edges(origin, data=True):
        removed_edges_G_reverse.append(edge)
    for edge in G_reinvert.in_edges(destination, data=True):
        removed_edges_G_reverse.append(edge)
    G_reinvert.remove_edges_from(removed_edges_G_reverse)

    # get all nodes which can be reached by origin and destination, i.e. which lower limit is not higher
    setOfFeasibleNodes, listOfReachableNodes = get_all_feasible_nodes(G, origin, G_reinvert, destination, dict_Solution_Parameters)

    # add back removed edges
    G.add_edges_from(removed_edges_G)
    G_reinvert.add_edges_from(removed_edges_G_reverse)

    # quick check for plausiblity
    inputProcessed = check_plausibility_of_determined_Nodes(setOfFeasibleNodes, origin, destination,
                                                            listOfReachableNodes, G, dict_Solution_Parameters)

    return setOfFeasibleNodes


def get_all_feasible_nodes(G, origin, G_reinvert, destination, dict_Solution_Parameters):
    listOfReachableNodes = {}

    # In order to reduce runtime first to a broader search and then a refined search
    # The broader search looks at the union of all points, reachable from origin and destination with at most half the
    # determined solutikon value. Also both sets, are determined by a single thread before their union is build up.
    thread_origin = myThread(G, origin, (dict_Solution_Parameters['Path_Disutility'] + 0.01)/2 ,listOfReachableNodes, 'origin', 'weight')
    thread_destination = myThread(G_reinvert, destination, (dict_Solution_Parameters['Path_Disutility'] + 0.01)/2 ,listOfReachableNodes, 'destination', 'weight')
    thread_origin.start()
    thread_destination.start()
    thread_origin.join()
    thread_destination.join()

    setOfFeasibleNodes = set(listOfReachableNodes['origin'].keys()).union(set(listOfReachableNodes['destination'].keys()))

    # With the broader search, we can quickly reduce the graph. This allows for a quicker refined search, were the actual lower limit is approximated
    # and nodes are deleted if possible
    listOfReachableNodes = {}
    thread_origin = myThread(G, origin, (dict_Solution_Parameters['Path_Disutility'] + 0.01) ,listOfReachableNodes, 'origin', 'weight', reduceGraph=True, setOfFeasibleNodes=setOfFeasibleNodes)
    thread_destination = myThread(G_reinvert, destination, (dict_Solution_Parameters['Path_Disutility'] + 0.01) ,listOfReachableNodes, 'destination', 'weight',reduceGraph=True, setOfFeasibleNodes=setOfFeasibleNodes)
    thread_origin.start()
    thread_destination.start()
    thread_origin.join()
    thread_destination.join()

    # get intersection of both list, to first remove any node of which the path from origin or to destination already disqualifies the node
    nodesFromOrigin = set(listOfReachableNodes['origin'].keys())
    nodesToDestination = set(listOfReachableNodes['destination'].keys())
    setOfFeasibleNodes = nodesFromOrigin.intersection(nodesToDestination)
    nodesToBeRemoved = set()

    # remove any node, with higher limit than solution value
    for node in setOfFeasibleNodes:
        distanceToNode = listOfReachableNodes['origin'][node]
        distanceFromNode = listOfReachableNodes['destination'][node]
        if (dict_Solution_Parameters['Path_Disutility'] + 0.01) <= (
                distanceToNode + distanceFromNode):  # give some headway for numerical issues
            nodesToBeRemoved.add(node)

    setOfFeasibleNodes = setOfFeasibleNodes.difference(set(nodesToBeRemoved))

    return setOfFeasibleNodes, listOfReachableNodes


def resetPunishment(G, change_punisher, instanceNetwork):
    for (i,j,m) in instanceNetwork.arcs_for_refined_search:
        G[i][j]['refined_search_punishment'] = float(G[i][j]['weight']) + change_punisher


def check_plausibility_of_determined_Nodes(setOfFeasibleNodes, origin, destination, listOfReachableNodes,
                                           G, dict_Solution_Parameters) -> bool():
    plausible = True
    # check if origin and destination are in the set of feasible nodes
    if origin not in setOfFeasibleNodes and destination not in setOfFeasibleNodes:
        print('there is a problem')
        plausible = False
    # check if all nodes of identified solution tour are in the reduced Graph
    for fromNode, toNode in dict_Solution_Parameters['Tour']:
        if toNode not in setOfFeasibleNodes:
            print(dict_Solution_Parameters['Tour'])
            print(nx.dijkstra_path_length(G, source=origin, target=destination))
            print(dict_Solution_Parameters['Path_Disutility'])
            print(nx.dijkstra_path_length(G, source=origin, target=toNode))
            print(listOfReachableNodes['origin'][toNode])
            print(nx.dijkstra_path_length(G, source=toNode, target=destination))
            print(listOfReachableNodes['destination'][toNode])
            print(toNode)
            print('there is a problem!')
            raise SyntaxError

    if not plausible:
        print('syntaxerror')
        raise SyntaxError

    return plausible


def increase_cost_on_change_arcs(G, changePunisher, arcs_of_connectionLayer: list(), firstRun):
    try:
        if firstRun:
            for (fromNode, toNode, m) in arcs_of_connectionLayer:
                G[fromNode][toNode]['weightWithChangePunishment'] = float(G[fromNode][toNode]['weight']) + changePunisher
        else:
            for (fromNode, toNode, m) in arcs_of_connectionLayer:
                G[fromNode][toNode]['weightWithChangePunishment'] += changePunisher

    except ValueError:
        print('There is no mode in the Graph!')
        raise SystemExit


def decrease_cost_on_change_arcs(G, arcs: list(), usedModes: list(), changePunisher):
    try:
        changed_edges = []
        for (fromNode, toNode, m) in arcs:
            # if (fromNode, toNode) in G.edges():
            toMode = toNode.split('+')[1]
            if toMode in usedModes:
                G[fromNode][toNode]['refined_search_punishment'] += (4 * G[fromNode][toNode]['cur_punishment'])
                changed_edges.append((fromNode, toNode))
        if changePunisher != 0:
            for (fromNode, toNode, m) in arcs:
                G[fromNode][toNode]['refined_search_punishment'] += changePunisher
                changed_edges.append((fromNode, toNode))
        return changed_edges
    except ValueError:
        print('There is no mode in the Graph!')
        raise SystemExit

def get_back_original_cost_on_change_arcs(G, instanceNetwork, changePunisher, forbidModes):
    if changePunisher <= 0.01:
        # only reset drive and bike
        for i,j,m in instanceNetwork.arcs_for_refined_search:
            G[i][j]['refined_search_punishment'] = G[i][j]['weight']
    for i,j in forbidModes:
        G[i][j]['refined_search_punishment'] = G[i][j]['weight']


def get_solution_parameters(G, shortestPathTour, modesConsideredNotPT, parameters):
    dict_solution_parameters = {'Number_of_Changes': 0, 'PublicTransport': 0, 'Bike': 0, 'Path_Disutility': 0,
                                'Tour': shortestPathTour, 'Feasibility': True}
    pathDisutility = 0
    for i, j in shortestPathTour:
        pathDisutility += G[i][j]['arc_disutility']
        fromMode = i.split('+')[1]
        toMode = j.split('+')[1]
        if fromMode != toMode:
            dict_solution_parameters['Number_of_Changes'] += 1
        if fromMode not in modesConsideredNotPT or toMode not in modesConsideredNotPT:
            dict_solution_parameters['PublicTransport'] = 1
        if fromMode == 'bike' or toMode == 'bike':
            dict_solution_parameters['Bike'] = 1
    pathDisutility += parameters['beta'][3] * (
            dict_solution_parameters['PublicTransport'] * parameters['fixCost_PublicTransport'] + parameters[
        'fixCost_Bike'] * dict_solution_parameters['Bike'])
    dict_solution_parameters['Path_Disutility'] = pathDisutility
    solution_status = 'feasible'
    if not dict_solution_parameters['Number_of_Changes'] <= parameters['maxNumber_of_Changes']:
        solution_status = 'infeasible'
        dict_solution_parameters['Feasibility'] = False
    print('calculated ' + solution_status + ' solution, obj. value: ' + str(pathDisutility))
    return dict_solution_parameters

def add_mode_type_to_edges(dict_of_solution: {}):
    dict_of_solution_updated = dict_of_solution
    dict_of_solution_updated['TourWithModes'] = list()
    for (i,j) in dict_of_solution['Tour']:
        fromMode = i.split('+')[1]
        toMode = j.split('+')[1]
        if fromMode == toMode:
            dict_of_solution_updated['TourWithModes'].append((i,j,fromMode))
        else:
            dict_of_solution_updated['TourWithModes'].append((i, j, 'connectionLayer'))
    return dict_of_solution_updated


def write_Solution_to_CSV(file_path: str(), beta: list(), dictOfSolution: dict(), origin, destination):
    if networkIsAccessibleAtDisk(file_path):
        with open(file_path, 'a+') as infoFile:
            fileWriter = csv.writer(infoFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fileWriter.writerow([dictOfSolution['maxNumber_of_Changes'],beta[0], beta[1], beta[2], beta[3], beta[4],
                                 origin, destination, dictOfSolution['Disutility'], dictOfSolution['inVehicleTime'],
                                 dictOfSolution['inVehicleTimePT'], dictOfSolution['inVehicleTimeDrive'],
                                 dictOfSolution['inVehicleTimeBK'],
                                 dictOfSolution['walkingTime'], dictOfSolution['waitingTime'], dictOfSolution['cost'],
                                 len(dictOfSolution['w']), dictOfSolution['preprocessingTime'], dictOfSolution['optimizationTime'],dictOfSolution['Gap']])
    else:
        with open(file_path, mode='w') as infoFile:
            fileWriter = csv.writer(infoFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fileWriter.writerow(['maxChanges', 'beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5',
                                 'origin', 'destination', 'Disutility', 'inVehicleTime', 'ptTime', 'taxiTime',
                                 'bikeTime', 'walkingTime', 'waitingTime', 'cost', 'numberOfChanges', 'preprocessingTime', 'optimizationTime', 'Gap'])
            fileWriter.writerow([dictOfSolution['maxNumber_of_Changes'],beta[0], beta[1], beta[2], beta[3], beta[4],
                                 origin, destination, dictOfSolution['Disutility'], dictOfSolution['inVehicleTime'],
                                 dictOfSolution['inVehicleTimePT'], dictOfSolution['inVehicleTimeDrive'],
                                 dictOfSolution['inVehicleTimeBK'],
                                 dictOfSolution['walkingTime'], dictOfSolution['waitingTime'], dictOfSolution['cost'],
                                 len(dictOfSolution['w']), dictOfSolution['preprocessingTime'], dictOfSolution['optimizationTime'],dictOfSolution['Gap']])


def get_dict_of_constraints(modifiedEdgeListUpdated, modesConsideredPublicTransport, shortest_path_constraints) -> dict():
    """
    Function to determine all domains for each constrained.
    """
    dict_of_constraint = {'shortest_path': shortest_path_constraints, 'bike_indicator': {'entering': dict(), 'leaving': dict()},
                          'publicTransport_indicator': {'entering': dict(), 'leaving': dict()}}
    for (i, j, m) in modifiedEdgeListUpdated.keys():
        if m == 'connectionLayer':
            toMode = j.split('+')[1]
            if toMode == 'bike':
                dict_of_constraint['bike_indicator']['entering'][(i, j, m)] = (i, j, m)
            if toMode in modesConsideredPublicTransport:
                dict_of_constraint['publicTransport_indicator']['entering'][(i, j, m)] = (i, j, m)
            fromMode = i.split('+')[1]
            if fromMode == 'bike':
                dict_of_constraint['bike_indicator']['leaving'][(i, j, m)] = (i, j, m)
            if fromMode in modesConsideredPublicTransport:
                dict_of_constraint['publicTransport_indicator']['leaving'][(i, j, m)] = (i, j, m)

    return dict_of_constraint


def remove_unnecessary_edges(modifiedEdgeListUpdated, origin, destination):
    """
    Function to remove any node, which is only connected to one other node and thus is unncessary. This is the case,
    because we have only postive edge weights. And thus, visiting the node to then just go back the node you came from,
    is pointless and against the shortest path constrained.
    """
    dict_of_constraint = {'shortest_path': dict()}
    for (i, j, m) in modifiedEdgeListUpdated.keys():
        # from_arc
        if j in dict_of_constraint['shortest_path'].keys():
            dict_of_constraint['shortest_path'][j]['arcEnteringNode'].append((i, j, m))
        else:
            dict_of_constraint['shortest_path'][j] = {'arcEnteringNode': [(i, j, m)], 'arcLeavingNode': []}
        # to_arc
        if i in dict_of_constraint['shortest_path'].keys():
            dict_of_constraint['shortest_path'][i]['arcLeavingNode'].append((i, j, m))
        else:
            dict_of_constraint['shortest_path'][i] = {'arcEnteringNode': [], 'arcLeavingNode': [(i, j, m)]}


    # and remove the edges which are pointless e.g. going back an so on
    found_nodes_to_remove = True
    counter = 0
    start_time = time.time()
    length_entering = int()
    length_leaving = int()
    while found_nodes_to_remove:
        remove_nodes = []
        remove_arcs = {}
        test_counter = 0
        for node in dict_of_constraint['shortest_path'].keys():
            length_entering = len(dict_of_constraint['shortest_path'][node]['arcEnteringNode'])
            if length_entering == 1:
                length_leaving = len(dict_of_constraint['shortest_path'][node]['arcLeavingNode'])
                if length_leaving == 1:
                    (a, b, c) = dict_of_constraint['shortest_path'][node]['arcEnteringNode'][0]
                    (d, e, f) = dict_of_constraint['shortest_path'][node]['arcLeavingNode'][0]
                    if a == e:
                        remove_nodes.append(node)
                        counter += 1
                        if a not in remove_arcs.keys():
                            remove_arcs[a]= {'leaving': [], 'entering': []}
                        remove_arcs[a]['leaving'].append((a,b,c))
                        remove_arcs[a]['entering'].append((d,e,f))

        remove_nodes = set(remove_nodes)

        if len(remove_nodes) <= 0:
            found_nodes_to_remove = False
            break
        for node in remove_nodes:
            del dict_of_constraint['shortest_path'][node]
        for node in remove_arcs.keys():
            if node not in remove_nodes:
                for arc in remove_arcs[node]['entering']:
                    dict_of_constraint['shortest_path'][node]['arcEnteringNode'].remove(arc)
                    del modifiedEdgeListUpdated [arc]
                for arc in remove_arcs[node]['leaving']:
                    dict_of_constraint['shortest_path'][node]['arcLeavingNode'].remove(arc)
                    del modifiedEdgeListUpdated[arc]
                counter += recursive_remove_of_node(dict_of_constraint['shortest_path'], node, modifiedEdgeListUpdated)


    return dict_of_constraint['shortest_path']


def recursive_remove_of_node(ordered_edges, node, edges):
    counter = 0
    if ordered_edges[node]['arcEnteringNode']:
        if len(ordered_edges[node]['arcEnteringNode']) == 1:
            if len(ordered_edges[node]['arcLeavingNode']) == 1:
                (a, b, c) = ordered_edges[node]['arcEnteringNode'][0]
                (d, e, f) = ordered_edges[node]['arcLeavingNode'][0]
                if a == e:
                    if a in ordered_edges.keys():
                        ordered_edges[a]['arcEnteringNode'].remove((d,e,f))
                        del edges[(d,e,f)]
                        ordered_edges[a]['arcLeavingNode'].remove((a,b,c))
                        del edges[(a,b,c)]
                        del ordered_edges[node]
                        counter += 1
                        counter += recursive_remove_of_node(ordered_edges,a, edges)
    return counter

def get_feasible_nodes(G, source, cutoff, return_dict, key):
    return_dict[key] = nx.single_source_dijkstra_path_length(G, source, cutoff=cutoff)

class myThread (threading.Thread):

    def __init__(self, G, source, cutoff, return_dict, key, weight, reduceGraph = False, setOfFeasibleNodes = list()):
        threading.Thread.__init__(self)
        self.G = G
        self.source = source
        self.cutoff = cutoff
        self.return_dict = return_dict
        self.key = key
        self.weight = weight
        self.reduceGraph = reduceGraph
        self.setOfFeasibleNodes = setOfFeasibleNodes


    def run(self):
        if self.reduceGraph:
            self.G = self.G.subgraph(self.setOfFeasibleNodes)
        self.return_dict[self.key] = nx.single_source_dijkstra_path_length(self.G, self.source, cutoff=self.cutoff, weight= self.weight)


def callback_for_cuts(model, where):

    # add cut, derived from model formulation, i.e. the variable Pt and BK are determined by the changes to the respective mode (Big-M Formulation)
    # hower both variables also need to be gigger than any edge with a mode belonging to them. If any edge with an value higher than PT or BK is determined
    # we can add this as a cut
    if where == gb.GRB.Callback.MIPNODE:
        status = model.cbGet(gb.GRB.Callback.MIPNODE_STATUS)
        if status == gb.GRB.OPTIMAL:
            list_of_non_PT_modes = ['walk', 'bike', 'drive', 'connectionLayer']
            dict_of_used_edges = {}
            # get all used edges, i.e. edges with value > 0
            for edge, solution_value in model.cbGetNodeRel(model._vars).items():
                if solution_value > 0.001:
                    dict_of_used_edges[edge] = solution_value
            dict_of_public_cuts = {}
            highest_value_bike = -1.0
            bike_arc = tuple()

            for edge, attr in dict_of_used_edges.items():
                # determine the edge with the highest value for each mode of PT
                if edge[2] not in list_of_non_PT_modes:
                    if edge[2] in dict_of_public_cuts.keys():
                        if dict_of_public_cuts[edge[2]][1] < attr:
                            # only for those edges that are still in the model (edge was might in callback to reduce model size)
                            if edge in model._linksNode:
                                dict_of_public_cuts[edge[2]] = (edge, attr)

                    else:
                       dict_of_public_cuts[edge[2]] = (edge, attr)
                # determine the edge with highest value (and thus cut) for BK
                elif edge[2] == 'bike':
                    if attr >= highest_value_bike:
                        highest_value_bike = attr
                        bike_arc = edge

            # As the edges are linked by the shortest path constrained we only need to add one cut per mode, eg. edge bus 52 <= PT and edge tram 17 <= PT

            # add cuts for PT
            for key_cut, cut in dict_of_public_cuts.items():
                # cut only makes sense, if edge value is really than higher PT
                if cut[1] > (model.cbGetNodeRel(model.getVarByName('PT'))+0.01):
                    pt_arc = cut[0]
                    var_name = 'y[' + str(pt_arc[0]) + ',' + str(pt_arc[1]) + ',' + str(pt_arc[2]) + ']'
                    left_hand_side = gb.LinExpr()
                    left_hand_side += model.getVarByName(var_name)
                    right_hand_side = gb.LinExpr()
                    right_hand_side += model.getVarByName('PT')
                    model.cbCut(lhs= left_hand_side, sense= gb.GRB.LESS_EQUAL, rhs = right_hand_side )
                    print('Added cut ' + str(var_name))

            # add cuts for BK
            # cut only makes sense, if edge value is really than higher BK
            if highest_value_bike > (model.cbGetNodeRel(model.getVarByName('BK'))+0.01) and len(bike_arc) != 0:
                var_name = 'y[' + str(bike_arc[0]) + ',' + str(bike_arc[1]) + ',' + str(bike_arc[2]) + ']'
                left_hand_side = gb.LinExpr()
                left_hand_side += model.getVarByName(var_name)
                right_hand_side = gb.LinExpr()
                right_hand_side += model.getVarByName('BK')
                model.cbCut(lhs=left_hand_side, sense=gb.GRB.LESS_EQUAL, rhs=right_hand_side)
                print('Added cut ' + str(var_name))

    # custom function, when gurobi identifies a new feasible solution
    if where == gb.GRB.Callback.MIPSOL:

        # get solution value of new feasible solution
        model._best = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBST)

        # check if new solution is significantly better than previous one
        if (model._preprocessingSolution['Path_Disutility'] / 1.10) >= model._best:
            if time.time() - model._time  >= 20:
                # when better, reduce graph size as in preprocessing and resart (terminate old)
                model._use_better_solution = True
                model.terminate()


def update_gurobi_model(model, linksNode):
    # set new best solution
    model._preprocessingSolution['Path_Disutility'] = model._best

    # determine all reachable nodes, as in preprocessing
    feasibleNodes, listOfReachableNodes = get_all_feasible_nodes(model._graphInstance.Graph_for_Preprocessing.subgraph(model._FeasibleNodes),
                                                                 model._graphOrigin, model._graphInstance.Graph_for_Preprocessing_Reverse.subgraph(model._FeasibleNodes),
                                           model._graphDestination,
                                           model._preprocessingSolution)

    # remove all edge variables, who contain a node, that is no part of the reduced Graph
    node_remain = gb.tuplelist()
    remove_link = []
    for i, j, m in linksNode:
        var_name = 'y[' + str(i) + ',' + str(j) + ',' + str(m) + ']'
        if i not in feasibleNodes:
            remove_link.append(var_name)
        else:
            if j not in feasibleNodes:
                remove_link.append(var_name)
            else:
                node_remain.append((i, j, m))

    print('Removed ' + str(len(remove_link)) + ' variables based on the new identified feasible solution')

    # remove links
    for link in remove_link:
        model.remove(model.getVarByName(link))

    return node_remain




