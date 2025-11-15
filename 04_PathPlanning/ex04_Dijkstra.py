import numpy as np
import math
import matplotlib.pyplot as plt
import random
import heapq
from map_2 import map

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True
    
    def __lt__(self, other):
        return self.f < other.f
                
def get_action():
    
    diagCost = np.sqrt(2)
    action_set = [[ 0, 1, 1],
                  [ 0,-1, 1],
                  [ 1, 0, 1],
                  [-1, 0, 1],
                  [ 1, 1, diagCost],
                  [ 1,-1, diagCost],
                  [-1, 1, diagCost],
                  [-1,-1, diagCost]]

    return action_set

def collision_check(obstacle_set, pos):
    return pos in obstacle_set

def dijkstra(start, goal, map_obstacle):
    obstacle_set = set(zip(map_obstacle[0], map_obstacle[1]))
    
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    
    open_list = []
    closed_list = set()
    
    heapq.heappush(open_list, (start_node.f, start_node))
    
    while open_list:

        # Find node with lowest cost
        cost, cur_node = heapq.heappop(open_list)
        
        # If goal, return optimal path
        if cur_node == goal_node:
            path = []
            while cur_node is not None:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]
        
        # If not goal, move from open list to closed list
        closed_list.add(cur_node.position)

        action_set = get_action()
        for (dx, dy, cost) in action_set:

            new_node = Node(cur_node, (cur_node.position[0] + dx, cur_node.position[1] + dy))
            
            # If collision expected, do nothing
            isCollision = collision_check(obstacle_set, new_node.position)
            if isCollision:
                continue
            
            # If already in closed list, do nothing
            if new_node.position in closed_list:
                continue
            
            new_node.f = cur_node.f + cost
            
            # If a better path already exists, skip
            skip_flag = False
            for _, open_node in open_list:
                if new_node == open_node and new_node.f >= open_node.f:
                    skip_flag = True
                    break
            
            if skip_flag:
                continue
            
            # If not in closed list, update open list
            heapq.heappush(open_list, (new_node.f, new_node))
            
        # show graph
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.1)
                

def main():

    start, goal, omap = map()

    if show_animation == True:
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k',  markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = dijkstra(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()


if __name__ == "__main__":
    main()

    

