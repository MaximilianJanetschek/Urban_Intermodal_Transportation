from Utilities.Data_Retrieval import *
from Utilities.Requests import *
from Utilities.Parameters import *

# prepare parameters
CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                            maxNumber_of_Changes=4, beta=[1, 2, 3, 12, 2.5], waiting_time_bike=4, waiting_time_drive=4)


# generate a Instance - see class Data Retrieval for detail
BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])  # also possible drive and bike
# BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

# get requests
initializeRequests()
requests = getNumberRequests(BerlinInstance, 100)


testConstraint = [5,6]
totalNumber = len(testConstraint) * len(requests)
counter = 1

for constraint in testConstraint:
    CaseParameters.dictOfParameters['maxNumber_of_Changes'] = constraint
    BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)

    for request in requests:
        try:
            origin_point = (request.get('fromLat'), request.get('fromLon'))
            destination_point = (request.get('toLat'), request.get('toLon'))
            tourMulti = multi_mode_optimization_in_Arc_fromulation(origin_point, destination_point, BerlinInstance,
                                                                   CaseParameters.dictOfParameters)
        except ValueError:
            print("this request is infeasible")
        print('Request ' + str(counter) + ' out of ' + str(totalNumber) + ' is calculated')
        counter += 1
