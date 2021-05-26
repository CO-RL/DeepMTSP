import os
import argparse
import numpy as np
import scipy.sparse
import utilities
from itertools import combinations


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

def generate_MTSP(random, filename, n_customers,m_salesman):
    """
    Generate a MTSP problem following
    
        Bektas, T.: The multiple traveling salesman problem: an overview of 
    formulations and solution procedures. Omega, 34(3) (2006), pp. 209–219..

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y

    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1

    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        cnt = cnt+1
        
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
            
        cnt = cnt+1
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
                  
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for j in range(n_customers)]) + f" = 1\n")
            
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f" = 1\n")
            
        
        # for i in range(1,n_customers):
        #     cnt = cnt+1
        #     file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")

        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i != j:
                    cnt = cnt+1
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} + u_{i+1} - u_{j+1} <= {n_customers-m_salesman-1} \n")   
       
        nvisit = int((n_customers-1)/m_salesman)+1
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} <= {nvisit}\n")

        
        file.write("\nBounds\n")
        # cnt = cnt+1
        # file.write(f" c{cnt}: u_{1} = 0\n")
        for i in range(0,n_customers):
            for j in range(0,n_customers):
                cnt = cnt+1
                if i == j:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} = 0\n")                      

        file.write("\nGenerals\n")

        for i in range(1,n_customers):
            file.write(f" u_{i+1} ")

        file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                if i != j:
                    file.write(f" x_{i+1}_{j+1} ")


def generate_MTSP1(random, filename, n_customers,m_salesman):
    """
    Generate a MTSP problem following
    
        Bektas, T.: The multiple traveling salesman problem: an overview of 
    formulations and solution procedures. Omega, 34(3) (2006), pp. 209–219..

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y

    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1

    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        cnt = cnt+1
        
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
            
        cnt = cnt+1
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
                  
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for j in range(n_customers)]) + f" = 1\n")
            
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f" = 1\n")
            
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")

        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i != j:
                    cnt = cnt+1
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} + u_{i+1} - u_{j+1} <= {n_customers-m_salesman-1} \n")   
       
        nvisit = int((n_customers-1)/m_salesman)+1
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} <= {nvisit}\n")

        
        # file.write("\nBounds\n")
        cnt = cnt+1
        file.write(f" c{cnt}: u_{1} = 0\n")
        for i in range(0,n_customers):
            for j in range(0,n_customers):
                cnt = cnt+1
                if i == j:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} = 0\n")

        file.write("\nBounds\n")

        file.write("\nGenerals\n")

        for i in range(1,n_customers):
            file.write(f" u_{i+1} ")

        file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                if i != j:
                    file.write(f" x_{i+1}_{j+1} ")

        file.write("\nEnd")

def generate_MTSP2(random, filename, n_customers,m_salesman):
    """
    Generate a MTSP problem following
    
        Bektas, T.: The multiple traveling salesman problem: an overview of 
    formulations and solution procedures. Omega, 34(3) (2006), pp. 209–219..

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y


    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1

    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        cnt = cnt+1
        
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
            
        cnt = cnt+1
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
                  
        cnt = cnt+1
        file.write(f" c{cnt}: u_{1} = 0\n")
        
        nvisit = n_customers/m_salesman +1 
        
        #print(n_customers,m_salesman,nvisit)
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            #file.write(f" c{cnt}: u_{i+1} <= {n_customers-1}\n")
            file.write(f" c{cnt}: u_{i+1} <= {nvisit-1}\n")
            

        for i in range(0,n_customers):
            for j in range(0,n_customers):
                cnt = cnt+1
                if i == j:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} = 0\n")
                else:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} >= 0\n")
                    cnt = cnt+1
                    file.write(f" c{cnt}: x_{i+1}_{j+1} <= 1\n")
                    
  
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for j in range(n_customers)]) + f" = 1\n")
            
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f" = 1\n")
            
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")
        
        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i == j:
                    continue
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} <= {n_customers-m_salesman-1} \n")
                else:
                    cnt = cnt+1
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} + u_{i+1} - u_{j+1} <= {n_customers-m_salesman-1} \n")
        
        file.write("\nBounds\n")
        #for i in range(n_customers):
        #    file.write(f" 1 <= u_{i+1} <= {n_customers}\n")
        
        #for i in range(n_customers):
        #    for j in range(n_customers):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nGenerals\n")
            
        #file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                file.write(f" x_{i+1}_{j+1} ")
        for i in range(n_customers):
            file.write(f" u_{i+1} ")
            
        #file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        file.write("\nEnd")

def generate_p_median(random, filename, n_customers, n_facilities, p):
    """
    Generate a p-median Problem following
    
    Originally Published:
        S. L. Hakimi. 1964. Optimum Locations of Switching Centers and 
        the Absolute Centers and Medians of a Graph. Operations Research.
        12 (3):450-459.
    Adapted from:
            -1-
        ReVelle, C.S. and Swain, R.W. 1970. Central facilities location.
        Geographical Analysis. 2(1), 30-42.
            -2-
        Toregas, C., Swain, R., ReVelle, C., Bergman, L. 1971. The Location
        of Emergency Service Facilities. Operations Research. 19 (6),
        1363-1373.
            - 3 -
        Daskin, M. (1995). Network and discrete location: Models, algorithms,
        and applications. New York: John Wiley and Sons, Inc.
    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    #rng = np.random.RandomState(1)
    
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    #customers = generate_candidate_sites(n_customers)
    #facilities = generate_candidate_sites(n_facilities)
    
    #trans_costs = distance_matrix(customers,facilities)
    
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)

    
    
    #print(trans_costs)
    
    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\n")
        file.write(" obj: "+"".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        
        #file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\nsubject to\n")
        
        # assignment constraints
        cnt = 0 
        for i in range(n_customers):
            cnt = cnt+1
            file.write(f" constraints{cnt}: " +"".join([f"+ 1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" = 1\n")

    
        cnt = cnt+1
        # facilty constraint
        file.write(f" constraints{cnt}: ")
        for j in range(n_facilities):
            #if j<1:
            #    file.write(f"y_{j+1}")
            #else :
            file.write(f" +1 y_{j+1}")
        file.write(f" = {p}\n")
                   
        # opening constraints
        for j in range(n_facilities):
            for i in range(n_customers):
                cnt = cnt+1
                file.write(f" constraints{cnt}: " + f" -1 x_{i+1}_{j+1}" + f"+1 y_{j+1} >= 0\n")
                #file.write(f" constraints{cnt}: " +"".join([f" -1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f"+1 y_{j+1} >= 0\n")

        # optional constraints for LP relaxation tightening

        #file.write("Bounds\n")
        #    for j in range(n_facilities):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")
        
        #file.write("".join([f" 0 <= y_{j+1} <=1 \n" for j in range(n_facilities)]))
        
        file.write("Bounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("".join([f" 0 <= y_{j+1} <=1 \n" for j in range(n_facilities)]))

        file.write("\nbinary\n")
        #for i in range(n_customers):
        #    for j in range(n_facilities):
        #        file.write(f" x_{i+1}_{j+1}\n")
        
        #file.write("Binaries\n")
        
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" x_{i+1}_{j+1}\n")
        
        file.write("".join([f" y_{j+1} \n" for j in range(n_facilities)]))
        file.write("End")

def generate_p_center(random, filename, n_customers, n_facilities, p):
    """
    Generate a p-center Problem following

    Originally Published:
        S. L. Hakimi. 1964. Optimum Locations of Switching Centers and
        the Absolute Centers and Medians of a Graph. Operations Research.
        12 (3):450-459.
    Adapted from:
        Daskin, M. (1995). Network and discrete location: Models, algorithms,
        and applications. New York: John Wiley and Sons, Inc.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    p: int
        The number of facilities.
    """
    #rng = np.random.RandomState(1)

    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    #customers = generate_candidate_sites(n_customers)
    #facilities = generate_candidate_sites(n_facilities)
    #trans_costs = distance_matrix(customers,facilities)

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)

    #print(trans_costs)

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\n")

        file.write(" obj: "+f"1 W_{1}")


        file.write("\nsubject to\n")

        # assignment constraints
        cnt = 0
        for i in range(n_customers):
            cnt = cnt+1
            file.write(f" constraints{cnt}: " +"".join([f"+ 1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" = 1\n")


        cnt = cnt+1
        # facilty constraint
        file.write(f" constraints{cnt}: ")
        for j in range(n_facilities):
            #if j<1:
            #    file.write(f"y_{j+1}")
            #else :
            file.write(f" +1 y_{j+1}")
        file.write(f" = {p}\n")

        # opening constraints
        for j in range(n_facilities):
            for i in range(n_customers):
                cnt = cnt+1
                file.write(f" constraints{cnt}: " + f" -1 x_{i+1}_{j+1}" + f"+1 y_{j+1} >= 0\n")
                #file.write(f" constraints{cnt}: " +"".join([f" -1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f"+1 y_{j+1} >= 0\n")

        # minimize maximum constraints
        for j in range(n_facilities):
            cnt = cnt+1
            file.write(f" constraints{cnt}: "+"".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f"-1 W_{1}<=0\n" )


        file.write("bounds\n")
        file.write(f"0 <= W_{1} \n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("".join([f" 0 <= y_{j+1} <=1 \n" for j in range(n_facilities)]))
        #for i in range(n_customers):
        #    for j in range(n_facilities):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        #file.write("".join([f" 0 <= y_{j+1} <=1 \n" for j in range(n_facilities)]))

        file.write("binary\n")
        #for i in range(n_customers):
        #    for j in range(n_facilities):
        #        file.write(f" x_{i+1}_{j+1}\n")

        #file.write("Binaries\n")

        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" x_{i+1}_{j+1}\n")

        file.write("".join([f" y_{j+1} \n" for j in range(n_facilities)]))
        file.write("End")

def generate_MinMax_MTSP(random, filename, n_customers,m_salesman):
    """
    Generate a generate_MinMax_MTSP problem following
    
  Necula, R., Breaban, M., Raschip, M.: Tackling the Bi-criteria Facet of Multiple Traveling Salesman Problem with Ant Colony Systems, 27th International Conference on Tools with Artificial Intelligence (ICTAI), 9-11 November, Vietri sul Mare, Italy, pp. 873-880, 2015

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y

    #m_salesman = 3

    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1

    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write(f"1 W_{1}\n")
        
        
        #file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        
        
        for k in range(0,m_salesman):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}_{k+1}" for j in range(1,n_customers)]) + f" = {1}\n")
        
        for k in range(0,m_salesman):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}_{k+1}" for j in range(1,n_customers)]) + f" = {1}\n")
                  
        cnt = cnt+1
        file.write(f" c{cnt}: u_{1} = 0\n")
        
        nvisit = n_customers/m_salesman +1 
        
        #print(n_customers,m_salesman,nvisit)
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} <= {n_customers-1}\n")
            #file.write(f" c{cnt}: u_{i+1} <= {nvisit-1}\n")
            
        for i in range(0,n_customers):
            for j in range(0,n_customers):
                for k in range(0,m_salesman): 
                    cnt = cnt+1
                    if i == j:
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} = 0\n")
                    else:
                        continue
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} >= 0\n")
                        cnt = cnt+1
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} <= 1\n")
                    
  
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1}" for i in range(n_customers) for k in range(m_salesman)]) + f" = 1\n")
        
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1}" for j in range(n_customers) for k in range(m_salesman)]) + f" = 1\n")
            
        
        for j in range(1,n_customers):
            for k in range(0,m_salesman):
                cnt = cnt+1
                file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1} - 1 x_{j+1}_{i+1}_{k+1}" for i in range(n_customers) ]) + f" = 0\n")
        
        #for i in range(1,n_customers):
        #   cnt = cnt+1
        #    file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")
        
        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i == j:
                    continue
                # for k in range(m_salesman): 
                    #file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} <= {n_customers-m_salesman-1} \n")
                cnt = cnt+1
                    #file.write(f" c{cnt}: + u_{i+1} - u_{j+1} + {n_customers-m_salesman} "+"(" + "".join([f" + x_{i+1}_{j+1}_{k+1}" for k in range(n_customers)]) +")" + f"<={n_customers-m_salesman-1} \n")
                file.write(f" c{cnt}: + u_{i+1} - u_{j+1}  "+ "".join([f" + {n_customers-m_salesman} x_{i+1}_{j+1}_{k+1}" for k in range(n_customers)]) + f" <= {n_customers-m_salesman-1} \n")


        
        for k in range(m_salesman):
            cnt = cnt +1 
            file.write(f" c{cnt}:" + "".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}_{k+1}" for i in range(n_customers) for j in range(n_customers)]) + f"-1 W_{1} <= 0\n")
            
        
        
        file.write("\nBounds\n")
        file.write(f"0 <= W_{1} \n")
        #for i in range(n_customers):
        #    file.write(f" 1 <= u_{i+1} <= {n_customers}\n")
        
        #for i in range(n_customers):
        #    for j in range(n_customers):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nGenerals\n")
            
        #
        for i in range(n_customers):
            file.write(f" u_{i+1} ")
        
        file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                for k in range(m_salesman): 
                    file.write(f" x_{i+1}_{j+1}_{k+1} ")
                    
        #file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        file.write("\nEnd")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['MTSP2', 'MTSP1', 'minmax-mtsp', 'p_median'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utilities.valid_seed,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'MTSP2':
        # number_of_customers = 12
        # number_of_salesman = 3
        filenames = []
        ncustomerss = []
        nsalesmans = []

        # # train instances
        # n = 2000
        # lp_dir = f'data/instances/MTSP/train_{number_of_customers}_{number_of_salesman}'
        # print(f"{n} instances in {lp_dir}")
        # os.makedirs(lp_dir)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])   #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
        # ncustomerss.extend([number_of_customers] * n)
        # nsalesmans.extend([number_of_salesman] * n)

        # # validation instances
        # n = 400
        # lp_dir = f'data/instances/MTSP/valid_{number_of_customers}_{number_of_salesman}'
        # print(f"{n} instances in {lp_dir}")
        # os.makedirs(lp_dir)
        # filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        # ncustomerss.extend([number_of_customers] * n)
        # nsalesmans.extend([number_of_salesman] * n)


        # small transfer instances
        n = 10
        number_of_customers = 9
        number_of_salesman = 3
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # medium transfer instances
        number_of_customers = 12
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # big transfer instances   
        number_of_customers = 15
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # test instances
        number_of_customers = 18
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 30
        number_of_salesman = 5
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 40
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 50
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # actually generate the instances
        for filename, ncs, nsm in zip(filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MTSP2(rng, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")

    elif args.problem == 'MTSP1':
        # number_of_customers = 12
        # number_of_salesman = 3
        filenames = []
        ncustomerss = []
        nsalesmans = []


        # small transfer instances
        n = 10
        number_of_customers = 9
        number_of_salesman = 3
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # medium transfer instances
        number_of_customers = 12
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # big transfer instances   
        number_of_customers = 15
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # test instances
        number_of_customers = 18
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 30
        number_of_salesman = 5
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 40
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        number_of_customers = 50
        lp_dir = f'data/instances/{args.problem}/test_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # actually generate the instances
        for filename, ncs, nsm in zip(filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MTSP1(rng, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")

    elif args.problem == 'minmax-mtsp':
        number_of_customers = 9
        number_of_salesman = 3
        filenames = []
        ncustomerss = []
        nsalesmans = []


        # small transfer instances
        n = 100
        number_of_customers = 9
        number_of_salesman = 3
        lp_dir = f'data/instances/minmax-mtsp/transfer_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # medium transfer instances
        n = 100
        number_of_customers = 12
        lp_dir = f'data/instances/minmax-mtsp/transfer_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        # big transfer instances
        n = 100
        number_of_customers = 15
        lp_dir = f'data/instances/minmax-mtsp/transfer_{number_of_customers}_{number_of_salesman}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)


        # actually generate the instances
        for filename, ncs, nsm in zip(filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MinMax_MTSP(rng, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")

    elif args.problem == 'p_median':
        number_of_customers = 100
        number_of_facilities = 100
        p = 5
        filenames = []
        ncustomerss = []
        nfacilitiess = []
        ps = []

        # small transfer instances
        n = 20
        number_of_customers = 100
        number_of_facilities = 100
        lp_dir = f'data/instances/p_median/transfer_{number_of_customers}_{number_of_facilities}_{p}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ps.extend([p] * n)

        # medium transfer instances
        n = 20
        number_of_customers = 200
        lp_dir = f'data/instances/p_median/transfer_{number_of_customers}_{number_of_facilities}_{p}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ps.extend([p] * n)

        # big transfer instances
        n = 100
        number_of_customers = 400
        lp_dir = f'data/instances/p_median/transfer_{number_of_customers}_{number_of_facilities}_{p}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ps.extend([p] * n)

        # actually generate the instances 
        for filename, ncs, nfs, p in zip(filenames, ncustomerss, nfacilitiess, ps):
        
            print(f"  generating file {filename} ...")
            generate_p_median(rng, filename, n_customers=ncs, n_facilities=nfs, p=p)
        
        print("done.")
