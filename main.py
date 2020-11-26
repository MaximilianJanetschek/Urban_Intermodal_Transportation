from Utilities.Data_Retrieval import *
from Utilities.Requests import *
from Utilities.Parameters import *

# get requests
# initializeRequests()
requests = [ # format (lat, lon, requestID)
        [(52.48879649, 13.4076626),(52.4737456, 1345574917), 11],
         [(52.50556889, 13.36040337), (52.49159616, 13.45763595), 12],
          [(52.48157791, 13.44006092),(52.49012295, 13.40565835), 37],
          [(52.49437194, 13.34795722), (52.46564987, 13.30599839), 59],
          [(52.48382454, 13.39427417), (52.4255787, 13.48314298), 64],
          [(52.49298845, 13.32344544), (52.48958601, 13.31215569), 70],
          [(52.50659051, 13.30669561), (52.50799763, 13.4354731), 71],
          [(52.53382324, 13.41284736), (52.5541665, 13.40249121),81],
          [(52.50743075, 13.42767829),(52.52566639, 13.31604926), 92]
            ]
betaSets = [[1, 2, 3, 12, 2.5],[1, 10, 3, 12, 12.5],[1, 2, 3, 1, 2.5] ]


for i in betaSets:
    CaseParameters = Parameters(varCost_Taxi=0.0023, fixCostTaxi=3.9, fixCost_PublicTransport=2.9, fixCost_Bike=1.5,
                            maxNumber_of_Changes=4, beta=i, waiting_time_bike=4, waiting_time_drive=4)

    # generate a Instance - see class Data Retrieval for detail
    BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])  # also possible drive and bike
    BerlinInstance.generateMultiModalGraph(parameters=CaseParameters.dictOfParameters)
    BerlinDrawing = Drawing('Berlin', BerlinInstance)

    for request in requests:
        # run model
        tourMulti = multi_mode_optimization_in_Arc_fromulation(request[0], request[1], BerlinInstance,CaseParameters.dictOfParameters)

        # draw tour
        BerlinDrawing.createFoliumModeColoring(tourMulti[0], tourMulti[1], str(request[2]) + '-beta Set ' + str(i))


