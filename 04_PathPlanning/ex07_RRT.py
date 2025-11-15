import numpy as np
import matplotlib.pyplot as plt
from map_4 import map


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class RRT(object):
    def __init__(self, start, goal, space, obstacle_list, success_dist_thres=1.0):
        self.start_node = Node(start[0], start[1])  # node (x, y)
        self.goal_node = Node(goal[0], goal[1])  # node (x, y)
        self.space = space  # (min_x, max_x, min_y, max_y)
        self.obstacle_list = obstacle_list  # list of (x, y ,r)
        self.node_list = []

        # parameters
        self.max_iter = 5000
        self.goal_sample_rate = 0.1
        self.min_u = 1.0
        self.max_u = 3.0
        self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def plan(self, animation=False, ax=None, pause_s=0.001):
        self.node_list = [self.start_node]

        for i in range(self.max_iter):
            # Get random node
            rand_node = self.get_random_node()
            # Find neareast node
            nearest_node = self.find_nearest_node(self.node_list, rand_node)
            # Create new node
            u = self.stepsize*self.get_random_input(self.min_u, self.max_u)
            new_node = self.create_child_node(nearest_node, rand_node, u)
            
            # Collision check(Node, Path)
            if self.is_collide(new_node, self.obstacle_list):
                continue
            if self.is_path_collide(nearest_node, new_node, self.obstacle_list, self.collision_check_step):
                continue
            
            # Add to tree
            new_node.set_parent(nearest_node)
            self.node_list.append(new_node)
            
            # Realtime visualization (only when animation=True)
            if animation and ax is not None:
                ax.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], color='blue', linewidth=0.5, alpha=0.7)
                ax.plot(new_node.x, new_node.y, 'bo', markersize=2, alpha=0.3)
                ax.set_title(f"RRT exploring... / iter={i+1}")
                plt.draw()
                plt.pause(pause_s)
                
            # Goal check
            if self.check_goal(new_node, self.success_dist_thres):
                print(" [-] GOAL REACHED")
                path = self.backtrace_path(new_node)
                return path
            
        print(" [-] FAILED TO REACH GOAL")
        return None

    @staticmethod
    def is_same_node(node1, node2):
        return (node1.x == node2.x) and (node1.y == node2.y)

    def backtrace_path(self, node):
        current_node = node
        path = [current_node]
        reached_start_node = self.is_same_node(current_node, self.start_node)
        while not reached_start_node:
            current_node = current_node.parent
            path.append(current_node)
            reached_start_node = self.is_same_node(current_node, self.start_node)
        return path[::-1]

    def get_random_node(self):
        # set random node as goal node occasionally
        rand_x = np.random.uniform(self.space[0], self.space[1])
        rand_y = np.random.uniform(self.space[2], self.space[3])
        rand_node = Node(rand_x, rand_y)
        
        if np.random.rand() < 0.05:
            return self.goal_node
        else:
            return rand_node

    def check_goal(self, node, success_dist_thres):
        distance = np.sqrt((node.x - self.goal_node.x)**2 + (node.y - self.goal_node.y)**2)
        return distance <= success_dist_thres

    @staticmethod
    def create_child_node(nearest_node, rand_node, u):
        dx = rand_node.x - nearest_node.x
        dy = rand_node.y - nearest_node.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return False
        new_node = Node(nearest_node.x + u*dx/dist, nearest_node.y + u*dy/dist)
        return new_node

    @staticmethod
    def get_random_input(min_u, max_u):
        randomNumber = np.random.uniform(min_u, max_u, 1)
        return float(randomNumber)

    @staticmethod
    def find_nearest_node(node_list, rand_node):
        distance_list = [ (node.x - rand_node.x)**2 + (node.y - rand_node.y)**2 for node in node_list]
        min_index = np.argmin(distance_list)
        return node_list[min_index]

    @staticmethod
    def is_collide(node, obstacle_list):
        for (obs_x, obs_y, obs_r) in obstacle_list:
            if ((node.x - obs_x)**2 + (node.y - obs_y)**2) <= obs_r**2:
                return True
        return False

    @staticmethod
    def is_path_collide(node_from, node_to, obstacle_list, check_step=0.2):
        
        dx = node_to.x - node_from.x
        dy = node_to.y - node_from.y
        node_path = Node(node_from.x, node_from.y)
        step = check_step / np.sqrt(dx**2 + dy**2)
        
        t = 0.0
        while (t <= 1.0):
            x = node_from.x + t * dx
            y = node_from.y + t * dy
            node_path.x = x
            node_path.y = y
            
            if (RRT.is_collide(node_path, obstacle_list)):
                return True
            
            t += step
            
        return False


if __name__ == "__main__":
    start, goal, space, obstacle_list = map()
    success_dist_thres = 1.0

    # Map visualization
    fig, ax = plt.subplots()
    t = np.linspace(0, 2*np.pi, 30)
    for x, y, r in obstacle_list:
        ax.plot(x + r*np.cos(t), y + r*np.sin(t), 'k-')
    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'ro', markersize=8)
    ax.plot(goal[0] + success_dist_thres*np.cos(t), goal[1] + success_dist_thres*np.sin(t), 'g--')
    ax.set_xlim(space[0], space[1])
    ax.set_ylim(space[2], space[3])
    ax.grid(True)
    ax.set_aspect('equal')

    # RRT planning
    rrt = RRT(start, goal, space, obstacle_list, success_dist_thres)
    path = rrt.plan(animation=True, ax=ax, pause_s=0.001)

    # Result path visualization
    if ax is not None:
        for j in range(len(path)-1):
            n1, n2 = path[j], path[j+1]
            ax.plot([n1.x, n2.x], [n1.y, n2.y], 'r-', linewidth=2)
        for i, node in enumerate(rrt.node_list):
            ax.plot(node.x,node.y, 'bo', markersize=2, alpha=0.3)
            if node.parent is not None:
                ax.plot([node.parent.x, node.x], [node.parent.y, node.y], color='blue', linewidth=0.5, alpha=0.7)
            ax.set_title(f"RRT result / iter={i+1}" )
        plt.draw()

    plt.show()

