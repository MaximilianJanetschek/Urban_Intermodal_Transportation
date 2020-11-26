from Utilities.Data_Retrieval import *
from Utilities.Requests import *
from Utilities.Parameters import *
import time

# prepare parameters
CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                            maxNumber_of_Changes=4, beta=[1, 2, 3, 12, 2.5],  waiting_time_bike=4, waiting_time_drive=4)

# generate a Instance - see class Data Retrieval for detail
BerlinInstance = InstanceNetwork(place='Berlin, Germany',
                                 networks=['drive', 'walk', 'bike'])  # also possible drive and bike
BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

initializeRequests()

requests = getNumberRequests(BerlinInstance, 100)
start_time = time.time()
for i, request in enumerate(requests):
    try:
        origin_point = (request.get('fromLat'), request.get('fromLon'))
        destination_point = (request.get('toLat'), request.get('toLon'))
        tourMulti = multi_mode_optimization_in_Arc_fromulation(origin_point, destination_point, BerlinInstance,
                                                               CaseParameters.dictOfParameters)
    except ValueError:
        print("this request is infeasible")

    print("--- %s seconds ---" % round((time.time() - start_time), 2) + ' for ' + str(i + 1) + ' out of 100 requests')
