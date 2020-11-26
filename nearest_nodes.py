from Utilities.Requests import *
from Utilities.Data_Retrieval import *
from Utilities.Data_Manipulation import *

BerlinInstance = InstanceNetwork(place='Berlin, Germany', networks=['drive', 'walk', 'bike'])
requests = getNumberRequests(BerlinInstance, 100)

# determine ranking of nodes closest to city

center = {'y': 52.520008,'x' : 13.404954}
request_dist = {}
MapOfNetwork = folium.Map(location=(52.520008, 13.404954), zoom_start=15.3)
for i, request in enumerate(requests):
    origin = {'y': request.get('fromLat'), 'x':request.get('fromLon')}
    request_dist[i] = euclidean_distance(center, origin)
    folium.Marker(location=(origin['y'], origin['x']),
                  icon=folium.DivIcon(html=f"""<div style="color: {'deeppink'};">{i}</div>""")
                  ).add_to(MapOfNetwork)
MapOfNetwork.save('results/plots/graphConnectionTest.html')

request_rank = {}
for rank in range(0, 100):
    best = 100000000
    best_request = 0
    for request, dist in request_dist.items():
        if best > dist:
            best_request = request
            best = dist
    request_rank[best_request] = dist
    del request_dist[best_request]

print(request_rank)



