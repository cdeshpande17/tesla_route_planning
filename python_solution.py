# Solution:
    # build an adjacency list with all of the reachable nodes from a starting node
    # build a graph with a starting node
        # assign a weight based on:
            # If the next charger charges faster than the source, only charge the amount needed to reach that charger then go to that charger
            # If the next charger charges slower than the source, charge fully and then go to that charger
    # run dijkstras

from python_data import *
from math import radians, cos, sin, asin, sqrt
import time
from tqdm import tqdm

class Supercharger():
    def __init__(self, name, lat, lon, charging_speed):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.charging_speed = charging_speed
        self.set_of_adjacent_chargers = []

class Optimizer():
    def __init__(self):
        self.location_data = get_locations()
        self.chargers = {}
        self.name_to_node = {}
        self.graph = {}
        self.path = {}
        self.max_charge = 320
        self.speed = 105
        self.network_of_chargers(self.location_data)
        # for x,y in self.chargers.items():
        #     print(x, [j.name for j in y])
        #     print()
        self.adjacency_matrix = []
        self.build_potential_paths("East_Greenwich_RI", "Atlanta_GA")
        # [print(k,v) for k,v in self.graph.items()]
        self.shortest_path("East_Greenwich_RI", "Atlanta_GA")
        print(self.get_path("East_Greenwich_RI", "Atlanta_GA"))
        # print(self.calc_charging_times(self.get_path("Council_Bluffs_IA", "Cadillac_MI")))
        # print([x.name for x in self.name_to_node['Brooklyn_NY'].set_of_adjacent_chargers])
        # print()
        # print(list(self.graph['Brooklyn_NY'].keys()))
        # print()
        # print(self.graph)

    def great_circle_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers is 6356.752
        km = 6356.752 * c
        return km
    
    def network_of_chargers(self, locations):
        nodes = []
        for name, x, y, speed in locations:
            newNode = Supercharger(name, x, y, speed)
            self.name_to_node[name] = newNode
            nodes.append(newNode)
        for i,node in enumerate(nodes):
            for adjacent_node in nodes[i+1:]:
                distance_to_station = self.great_circle_distance(node.lon, node.lat, adjacent_node.lon, adjacent_node.lat)
                if (distance_to_station <= self.max_charge):
                    node.set_of_adjacent_chargers.append(adjacent_node)
                    adjacent_node.set_of_adjacent_chargers.append(node)

    def build_potential_paths(self, starting_node, ending_node):
        start = self.name_to_node[starting_node] # starting node
        end = self.name_to_node[ending_node] # ending node
        # y = mx + b
        m = (end.lon - start.lon) / (end.lat - start.lat)
        b = start.lon - (m * (start.lat))
        seen = set()
        reachable = [(start, self.max_charge, [start.name])] # queue
        while reachable:
            # print([(x[0].name, x[1]) for x in reachable])
            node, curr_charge, parents_eval = reachable.pop(0) #(A,320) --> (B,C,D) --> B (A,C,D) --> (C,D,E)
            # time.sleep(3)
            # print(parents_eval)
            seen.add(node.name)
            if node.name not in self.graph:
                self.graph[node.name] = {}
            for next_node in node.set_of_adjacent_chargers:
                second_m = -(1/m)
                second_b = next_node.lon - (second_m * next_node.lat)
                x_on_line = (second_b - b) / (m - second_m)
                y_on_line = (m * x_on_line) + b
                distance = self.great_circle_distance(y_on_line, x_on_line, next_node.lon, next_node.lat)
                if (next_node.name not in seen) and (next_node.name not in parents_eval) and (distance < 500):
                    distance_to_next = self.great_circle_distance(node.lon, node.lat, next_node.lon, next_node.lat)
                    travel_time = distance_to_next / self.speed
                    charge_needed = 0
                    if node.name == start.name:
                        charge_needed = 0 
                        curr_charge = self.max_charge - distance_to_next
                    elif next_node.charging_speed > node.charging_speed:    # min case (reach next node w/ no charge)
                        charge_needed = distance_to_next - curr_charge
                        curr_charge = 0
                    else:                                                   # max case
                        charge_needed = self.max_charge - curr_charge
                        curr_charge = self.max_charge - distance_to_next
                    if charge_needed < 0:
                        continue
                    charge_time = charge_needed / node.charging_speed
                    total_time = travel_time + charge_time
                    # print(next_node.name, "\t", distance_to_next, "\t", total_time)
                    #seen.add(next_node.name)
                    tmp = parents_eval + [x.name for x in node.set_of_adjacent_chargers]
                    reachable.append((next_node, curr_charge, tmp)) #(b,val)
                    self.graph[node.name][next_node.name] = (total_time, charge_time)

        # print(self.graph['Mountain_View_CA'])
        

    def shortest_path(self, source, target):
        spt = self.djikstra(self.graph, source)
        print(spt[target])

    def minValue(self, visited, distance):
        min_val = 0
        minimum = float('inf')
        for v in visited:
            if minimum > distance[v]:
                min_val = v
                minimum = distance[v]
        return min_val

    def djikstra(self, graph, source): 
        distance = {}
        visited = []
        
        distance[source] = 0
        for node in graph:
            if node != source:
                distance[node] = float('inf')
            visited.append(node)
        
        while len(visited) > 0:
            min_node = self.minValue(visited, distance)
            visited.remove(min_node)
            for neighbor in graph[min_node]:
                if neighbor not in visited:
                    continue
                alt = distance[min_node] + graph[min_node][neighbor][0]
                if alt < distance[neighbor]:
                    distance[neighbor] = alt
                    self.path[neighbor] = (min_node, graph[min_node][neighbor][1])
        return distance

    def get_path(self, start, ending):
        # print(self.path)
        gen_path = []
        locs = []
        times = []
        curr = ending # start at the end node
        while curr is not start: # build the path until the start value
            # print(gen_path)
            # time.sleep(1)
            gen_path.insert(0, curr)
            locs.insert(0, curr)
            if curr not in self.path: # if it gets to the node with no parent, then it is not possible
                return None
            gen_path.insert(0, self.path[curr][1])
            times.insert(0,self.path[curr][1])
            curr = self.path[curr][0] # go to the parent of the current
            

        
        gen_path.insert(0, start) # finally, insert the start node in
        locs.insert(0,start)
        print(self.calc_charging_times(locs))
        print(locs)
        return gen_path[:1] + gen_path[2:]

    def calc_charging_times(self, path):
        times = []
        for i in range(len(path)-1,0,-1):
            one = self.name_to_node[path[i-1]]
            two = self.name_to_node[path[i]]
            distance = self.great_circle_distance(one.lon, one.lat, two.lon, two.lat)
            times.insert(0,self.graph[path[i-1]][path[i]][0] - (distance / self.speed))
        return times
            

    # def calc_charging_times(self, path):
    #     cur_charge = 320
    #     times = []
    #     for i in range(len(path)-1):
    #         if i == 0:
    #             continue
    #         distance_to_prev = self.great_circle_distance(self.name_to_node[path[i]].lon, self.name_to_node[path[i]].lat , self.name_to_node[path[i-1]].lon, self.name_to_node[path[i-1]].lat)
    #         cur_charge -= distance_to_prev
    #         if self.name_to_node[path[i]].charging_speed > self.name_to_node[path[i+1]].charging_speed:
    #             times.append((320-cur_charge)/self.name_to_node[path[i]].charging_speed)
    #             cur_charge = 320
    #         else:
    #             distance = self.great_circle_distance(self.name_to_node[path[i]].lon, self.name_to_node[path[i]].lat , self.name_to_node[path[i+1]].lon, self.name_to_node[path[i+1]].lat)
    #             times.append((distance-cur_charge)/self.name_to_node[path[i]].charging_speed)
    #             cur_charge = distance
    #     return times



Optimizer()