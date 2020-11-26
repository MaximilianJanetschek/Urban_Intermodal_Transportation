class Parameters:
    """
    The class 'Parameter' is designed to hold every information regarding the specific network,
    like the different variable and fix costs for the respective mode or the parameter of the objective function
    in the mathematical model. There are no methods.
    """
    dictOfParameters = dict()
    varCost_Taxi = float()
    fixCost_Taxi = float()
    fixCost_PublicTransport = float()
    fixCost_Bike = float()
    waiting_time_bike = int()
    waiting_time_drive = int()
    maxNumber_of_Changes = float()
    beta = list()

    def __init__(self, varCost_Taxi: float(), fixCostTaxi: float(), fixCost_PublicTransport: float(),
                 fixCost_Bike: float(), maxNumber_of_Changes: int(), beta: [], waiting_time_drive: int(), waiting_time_bike:int()):
        self.dictOfParameters['varCost_Taxi'] = varCost_Taxi
        self.dictOfParameters['fixCost_Taxi'] = fixCostTaxi
        self.dictOfParameters['fixCost_PublicTransport'] = fixCost_PublicTransport
        self.dictOfParameters['fixCost_Bike'] = fixCost_Bike
        self.dictOfParameters['maxNumber_of_Changes'] = maxNumber_of_Changes
        self.dictOfParameters['beta'] = beta
        self.dictOfParameters['waiting_time_bike'] = waiting_time_bike
        self.dictOfParameters['waiting_time_drive'] = waiting_time_drive
        self.varCost_Taxi = varCost_Taxi
        self.fixCost_Taxi = fixCostTaxi
        self.fixCost_PublicTransport = fixCost_PublicTransport
        self.fixCost_Bike = fixCost_Bike
        self.maxNumber_of_Changes = maxNumber_of_Changes
        self.beta = beta
        self.waiting_time_bike = waiting_time_bike
        self.waiting_time_drive = waiting_time_drive

    def get_disutility_fix_cost_disutility(self, mode):
        if mode =='bike':
            return self.fixCost_Bike * self.beta [3]
        else:
            return self.fixCost_PublicTransport * self.beta [3]
