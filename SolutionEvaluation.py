import csv
from scipy.spatial import distance
import statistics
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as st


def getValues():
    values = []
    i = 1
    while i < 3:
        set = []
        with open(r'results/distances/results' + str(i) + '.CSV', newline='') as csvfile:
            solutionReader = csv.reader(csvfile, delimiter=';', quotechar='|')
            linecount = 0
            for row in solutionReader:
                if linecount == 0:
                    linecount += 1
                else:
                    set.append((row[0], row[1]))
        values.append(set)
        i +=1
    return values


def computeDistances(values):
    results = {}
    for i in range(0, len(values)):
        for j in range(1, len(values)):
            if i is not j:
                listdist = []
                for k in range(0, len(values[i])):
                    point1 = (float(values[j][k][0]),  float(values[j][k][1]))
                    point2 = (float(values[i][k][0]), float(values[i][k][1]))

                    # compute euclidean distance
                    dist = distance.euclidean(point1, point2)
                    listdist.append(dist)
                results[(i, j)] = listdist
    return results


def computeMeanDistances(distances):
    results = {}
    for pair in distances:
        results[pair] = (statistics.mean(distances.get(pair)), statistics.stdev(distances.get(pair)), st.t.interval(0.95, len(distances.get(pair))-1, loc = statistics.mean(distances.get(pair)),scale = st.sem(distances.get(pair))))
    return results


def printMeanAndStdvToCsv(result, path):
    with open(path, mode='w') as file:
        fileWriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileWriter.writerow(["betaset1", "betaset2", "mean distance", "std deviation", "confidence interval"])
        for pair in result:
            fileWriter.writerow([pair[0], pair[1], result.get(pair)[0], result.get(pair)[1], result.get(pair)[2]])

def printEuclideanDistance(distances, path):
    with open(path, mode='w') as file:
        fileWriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fileWriter.writerow(["Distance"])
        EuclideanDistance = list()
        for i in distances.keys():
            EuclideanDistance = distances[i]
        for j in range(0, len(EuclideanDistance)):
            fileWriter.writerow([EuclideanDistance[j],[]])



def plotValues(values, numberOfplottedRequests):
    #TODO connect dots with lines
    #number of requests
    numberOfRequests = len(values[0])
    #creating list of markers for requests
    forms = ["s", "v", "*", "s", "P"]
    colours = []
    for i in range(0, numberOfRequests):
        colours.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    j = 0
    for betaset in values:
        i = 0
        for timeCost in betaset:
            if i < numberOfplottedRequests:
                timeValues = [float(timeCost[0])]
                costValues = [float(timeCost[1])]
                plt.scatter(timeValues, costValues, color = colours[i], marker = forms[j], s = 30)
                i += 1
        j += 1
    plt.xlabel("Time")
    plt.ylabel("Cost")
    path = 'results/plots/TimeAndCost.png'
    plt.savefig(path, dpi=1000)
    plt.show()

def plotEuclideanValues(distances:dict):
    EuclideanDistance = list()
    for i in distances.keys():
        EuclideanDistance = distances[i]

    request = list(range(1,len(EuclideanDistance)+1))
    plt.scatter(request,EuclideanDistance, marker='o')

    plt.xlabel("Request")
    plt.ylabel("Euclidean Distance")
    path = 'results/plots/EuclideanDistance.png'
    plt.savefig(path, dpi=1000)
    plt.show()


values = getValues()
plotValues(values, 100)
distances = computeDistances(values)
plotEuclideanValues(distances)
meanDistances = computeMeanDistances(distances)
printMeanAndStdvToCsv(meanDistances, 'results/mean_and_stdv.csv')
printEuclideanDistance(distances,'results/EuclideanDistances.csv')
