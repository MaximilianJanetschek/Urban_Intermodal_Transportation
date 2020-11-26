import os

import networkx as nx
from Utilities.Data_Retrieval import *
import matplotlib.pyplot as plt
import osmnx as ox
import folium
from IPython.display import IFrame
from Utilities.General_Function import *


class Drawing:
    name = str()
    networkInstance = int()
    Graph = nx.MultiDiGraph()
    nodes_pos = dict()

    def __init__(self, name, networkInstance):
        self.name = name
        self.networkInstance = networkInstance
        self.generateDrawingOfNetwork()

    def generateDrawingOfNetwork(self):
        nodesList = list()
        self.nodes_pos = dict()
        print('generate Graph for Drawing')
        filePath = self.networkInstance.networksPath + str('setOfNodes.json')
        setOfNodes = getSavedDict(filePath)
        for node in setOfNodes:
            nodesList.append(node[0])
            self.nodes_pos[node[0]] = (node[1], node[2])
        self.Graph.add_nodes_from(nodesList)
        for edge in self.networkInstance.setOfEdges.keys():
            self.Graph.add_edge(edge[0], edge[1], key=0, mode=edge[2])

    def printGraph(self):
        print('Print Network')
        nx.draw_networkx_nodes(self.Graph, self.nodes_pos, node_size=0.0001, node_color='black')
        plt.axis('off')
        plot = plt
        plot.savefig("GraphOfBerlinMultiModalNetwork.png", dpi=1000)
        plt.show()

    def printGraphWithTour(self, tour: list(), name: ""):
        nodesOfTourInOrder = list()
        node_pos_tour = dict()
        reducedTourWithoutMode = list()
        for node in tour:
            reducedTourWithoutMode.append(node[0])
            node_pos_tour[node[0]] = self.nodes_pos[node[0]]
            nodesOfTourInOrder.append(node[0])
        node_color = ['red' if node in nodesOfTourInOrder else 'black' for node in self.Graph.nodes()]
        nx.draw_networkx_nodes(self.Graph, self.nodes_pos, node_size=1, cmap=plt.get_cmap('jet'),
                               node_color=node_color)
        nx.draw_networkx_nodes(self.Graph, nodelist=node_pos_tour.keys(), pos=node_pos_tour, node_size=1,
                               node_color='red')
        if name == "":
            path = "results/plots/GraphOfWithTour.png"
        else:
            path = 'results/plots/' + name + '.png'
        plt.axis('off')
        plot = plt
        plot.savefig(path, dpi=1000)
        plt.show()

    def printOwnGraph(self, G, tour, name: ""):
        nodesOfTourInOrder = list()
        node_pos_tour = dict()
        reducedTourWithoutMode = list()
        for node in tour:
            reducedTourWithoutMode.append(node[0])
            node_pos_tour[node[0]] = self.nodes_pos[node[0]]
            nodesOfTourInOrder.append(node[0])
        node_color = ['red' if node in nodesOfTourInOrder else 'black' for node in G.nodes()]
        nx.draw_networkx_nodes(G, self.nodes_pos, node_size=1, cmap=plt.get_cmap('jet'),
                               node_color=node_color)
        nx.draw_networkx_nodes(G, nodelist=node_pos_tour.keys(), pos=node_pos_tour, node_size=1, node_color='red')
        if name == "":
            path = "results/plots/GraphOfWithTour.png"
        else:
            path = 'results/plots/' + name + '.png'
        plt.axis('off')
        plot = plt
        plot.savefig(path, dpi=1000)

    def createFolium(self, tour: list()):
        nodesOfTourInOrder = list()
        node_pos_tour = dict()
        reducedTourWithoutMode = list()
        for node in tour:
            reducedTourWithoutMode.append(node[0])
            node_pos_tour[node[0]] = self.nodes_pos[node[0]]
            nodesOfTourInOrder.append(node[0])
        MapOfNetwork = folium.Map(location=(52.520008, 13.404954), zoom_start=15.3)
        for node in node_pos_tour.keys():
            folium.Marker(location=node_pos_tour[node],
                          icon=folium.Icon(color='red')).add_to(MapOfNetwork)
        MapOfNetwork.save('results/plots/graphTour.html')

    def createFoliumWithConnectionLayers(self):
        MapOfNetwork = folium.Map(location=(52.520008, 13.404954), zoom_start=15.3)
        for i, j, m in self.networkInstance.setOfEdges.keys():
            if m not in ['connectionLayer', 'drive', 'walk', 'bike']:
                folium.Marker(location=self.nodes_pos[i],
                              icon=folium.Icon(color='red')).add_to(MapOfNetwork)
        MapOfNetwork.save('results/plots/graphConnection.html')

    def createFoliumModeColoring(self, tour: list(), changes: list(), index):
        nodesOfTour = list()
        nodes_pos_tour = dict()
        print(tour)
        print(changes)
        for node in tour:
            nodes_pos_tour[(node[0], node[2])] = self.nodes_pos[node[0]]  # all i's in i, j, m
            nodesOfTour.append(node[0])
        nodes_pos_tour[(tour[-1][1], tour[-1][2])] = self.nodes_pos[
            tour[-1][1]]  # j in last node; destination which is not left again
        MapOfNetwork = folium.Map(location=(52.520008, 13.404954), zoom_start=15.3)
        for node in nodes_pos_tour.keys():
            if node[1] == 'walk':
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='lightgray')).add_to(MapOfNetwork)
            elif node[1] == 'connectionLayer':
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='lightgray')).add_to(MapOfNetwork)
            elif node[1] == 'drive':
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='orange')).add_to(MapOfNetwork)
            elif node[1] == 'bike':
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='purple')).add_to(MapOfNetwork)
            elif node[1][-3:] == '109':  # S-Bahn
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='green')).add_to(MapOfNetwork)
            elif node[1][-3:] == '400':  # U Bahn
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='blue')).add_to(MapOfNetwork)
            elif node[1][-3:] == '700':  # Bus
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='lightblue')).add_to(MapOfNetwork)
            else:  # tram
                folium.Marker(location=nodes_pos_tour[node], icon=folium.Icon(color='red')).add_to(MapOfNetwork)
        change_pos = dict()
        for change, fromMode, toMode in changes:
            change_pos[change] = self.nodes_pos[change]
        for change in change_pos.keys():
            folium.Marker(location=change_pos[change], icon=folium.Icon(color='white')).add_to(MapOfNetwork)
        savepath = 'results/plots/graphTourColored'
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        MapOfNetwork.save(savepath + '/' + str(index) + '.html')
