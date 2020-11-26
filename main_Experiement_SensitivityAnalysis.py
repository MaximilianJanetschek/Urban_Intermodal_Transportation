from Utilities.Data_Retrieval import *
from Utilities.Requests import *
from Utilities.Parameters import *
from Utilities.Solution_Procedure import *

run_CS = False
run_TS = True
run_PS = False

# beta experiment on Physical sensitive beta setup:
if run_CS:
    CSbetaset =[[1,2,3,12,2.5],
                [1,2,3,14,2.5],
                [1,2,3,17,2.5],
                [1,2,3,20,2.5],
                [1,2,3,22,2.5],
                [1,2,3,24,2.5],
                [1,2,3,27,2.5],
                [1,2,3,30,2.5],
                [1,2,3,34,2.5],
                [1,2,3,40,2.5],
                [1,2,3,44,2.5],
                [1,2,3,54,2.5],
                [1,2,3,60,2.5]]

    for i in range(0,len(CSbetaset)):
        if i >= 0:
            # prepare parameters
            CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                                        maxNumber_of_Changes=4, beta=CSbetaset[i], waiting_time_bike=4, waiting_time_drive=4)

            # generate a Instance - see class Data Retrieval for detail
            BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])  # also possible drive and bike
            BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

            # get requests
            initializeRequests()
            requests = getNumberRequests(BerlinInstance, 300)

            # run model
            total_number = len(requests)
            counter = 1
            start_time = time.time()
            for request_number in range(0, len(requests)):
                if request_number >= 271:
                    request = requests[request_number]
                    try:
                        origin_point = (request.get('fromLat'), request.get('fromLon'))
                        destination_point = (request.get('toLat'), request.get('toLon'))
                        tourMulti = multi_mode_optimization_in_Arc_fromulation(origin_point, destination_point, BerlinInstance,
                                                                           CaseParameters.dictOfParameters)
                        print(str(counter) +' out of ' + str(total_number) + ' requests are calculated')
                        print("--- %s seconds ---" % round((time.time() - start_time), 2))
                    except nx.NetworkXNoPath:
                        print("this does not work")
                print(str(counter) + ' out of ' + str(total_number) + ' requests are calculated')
                print("--- %s seconds ---" % round((time.time() - start_time), 2))
                counter += 1

        else:
            print('pass set '  + str(i) + ' as already generated')

# beta experiment on Physical sensitive beta setup:
if run_TS:
    TSbetaset = [[1,2,3,2,2.5],
                 # [1,2,3,1.75,2.5],
                 [1,2,3,1.7,2.5]]
    '''
    [[1,2,3,9,2.5],
            [1,2,3,6,2.5],
            [1,2,3,3,2.5],
            [1,2,3,1,2.5],
            [2,4,6,1,5],
            [3,6,9,1,7.5],
            [4,8,12,1,10],
            [5,10,15,1,12.5],
            [6,12,18,1,15]]
    '''
    for i in range(0,len(TSbetaset)):
        # set counter to last finished beta set, set to large number to skip physical test tun
        if i >= 0:
            # prepare parameters
            CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                                        maxNumber_of_Changes=4, beta=TSbetaset[i], waiting_time_bike=4, waiting_time_drive=4)

            # generate a Instance - see class Data Retrieval for detail
            BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])  # also possible drive and bike
            BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

            # get requests
            initializeRequests()
            requests = getNumberRequests(BerlinInstance, 300)

            # run model
            total_number = len(requests)
            counter = 1
            start_time = time.time()
            for request in requests:
                try:
                    origin_point = (request.get('fromLat'), request.get('fromLon'))
                    destination_point = (request.get('toLat'), request.get('toLon'))
                    tourMulti = multi_mode_optimization_in_Arc_fromulation(origin_point, destination_point, BerlinInstance,
                                                                       CaseParameters.dictOfParameters)
                    print(str(counter) +' out of ' + str(total_number) + ' requests are calculated')
                    print("--- %s seconds ---" % round((time.time() - start_time), 2))
                    counter += 1
                except nx.NetworkXNoPath:
                    print("this does not work")

# beta experiment on Physical sensitive beta setup:
if run_PS:
    PSbetaset =[[1,2,3,12,2.5],
                [1,3,3,12,3.75],
                [1,5,3,12,6.25],
                [1,6,3,12,7.5],
                [1,7,3,12,8.75],
                [1,8,3,12,10],
                [1,9,3,12,11.25],
                [1,10,3,12,12.5],
                [1,11,3,12,13.75],
                [1,12,3,12,15],
                [1,14,3,12,17.5],
                [1,16,3,12,20],
                [1,18,3,12,22.5]]

    for i in range(0,len(PSbetaset)):
        # set counter to last finished beta set, set to large number to skip physical test tun
        if i >= 0:
            # prepare parameters
            CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                                        maxNumber_of_Changes=4, beta=PSbetaset[i], waiting_time_bike=4, waiting_time_drive=4)

            # generate a Instance - see class Data Retrieval for detail
            BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])  # also possible drive and bike
            BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

            # get requests
            initializeRequests()
            requests = getNumberRequests(BerlinInstance, 300)

            # run model
            total_number = len(requests)
            counter = 1
            start_time = time.time()
            for request in requests:
                try:
                    origin_point = (request.get('fromLat'), request.get('fromLon'))
                    destination_point = (request.get('toLat'), request.get('toLon'))
                    tourMulti = multi_mode_optimization_in_Arc_fromulation(origin_point, destination_point, BerlinInstance,
                                                                       CaseParameters.dictOfParameters)
                    print(str(counter) +' out of ' + str(total_number) + ' requests are calculated')
                    print("--- %s seconds ---" % round((time.time() - start_time), 2))
                    counter += 1
                except nx.NetworkXNoPath:
                    print("this does not work")




