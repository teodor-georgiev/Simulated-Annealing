#!/usr/bin/python3

import numpy as np
import random
import math
import copy
import time

class Instance(object):
    def __init__(self, path):
        """
        Loads an instance.
        """
        with open(path, 'r') as fp:
            self.instance_name = path
            self.amountCustomers = int(fp.readline().split()[1])
            self.amountVehicles = int(fp.readline().split()[1])
            self.capacityVehicles = int(fp.readline().split()[1])             

            self.distances = np.zeros(shape=(self.amountCustomers + 1, self.amountCustomers + 1), dtype = int)

            fp.readline()            
            for ii in range(0,self.amountCustomers):
                for jj, element in enumerate([int(value) for value in fp.readline().split()]):
                    self.distances[ii][jj+ii+1] = element
                    self.distances[jj+ii+1][ii] = element

            fp.readline()
            self.demands = [0]
            self.demands.extend([int(value) for value in fp.readline().split()])



    def __repr__(self):
        return "{0.instance_name}\nAmount customers: {0.amountCustomers}\nAmount vehicles: {0.amountVehicles}\n" \
            "Vehicle capacity: {0.capacityVehicles}\n" \
            "Distance:\n{0.distances}\nDemands:\n{0.demands}".format(self)


class Simulated_Annealing(object):
    def __init__(self, inst: object, temperature: int, stop_temperature: int, iterations: int, cooling_rate: float) -> None:
        """
        Initializes the simulated annealing algorithm parameters and the instance.

        Parameters
        ----------
        inst : object
            Object of type Instance holding the instance data.
        temperature : int
            Starting temperature of the simulated annealing algorithm.
        stop_temperature : int
            Optional parameter, if set, the simulated annealing algorithm will stop when the temperature reaches this value.
        iterations : int
            Maximum number of iterations of the simulated annealing algorithm. 
        cooling_rate : float
            Coefficient of the cooling rate of the simulated annealing algorithm.
        """
        
        self.instance = inst
        self.temperature = temperature
        self.iterations = iterations
        self.stop_temperature = stop_temperature
        self.cooling_rate = cooling_rate
        
        self.customers = inst.amountCustomers
        self.vehicles = inst.amountVehicles
        self.vehicle_capacity = inst.capacityVehicles
        self.distance_matrix = inst.distances
        self.customers_demand = inst.demands
        
        # Masking the diagonal of the distance matrix
        self.distance_matrix= np.ma.masked_array(self.distance_matrix, self.distance_matrix < 1)
        
        self.best_solution = None
        self.best_objective = 10000000
        self.current_iteration = iterations
    
    def print_values(self,instance):
        print(self.vehicles)
        return print("Loaded instance: {0}".format(instance))
    
    def calc_objective(self,solution: dict)-> int:
        """
        Returns the objective function value of the given solution.

        Parameters
        ----------
        solution : dict
            Dictionary containing the solution to the vehicle routing problem.

        Returns
        -------
        int
            Objective function value of the given solution.
        """

        objective = 0
        for i in solution:
            for j in range (len(solution[i])-1):
                objective += self.distance_matrix[solution[i][j]][solution[i][j+1]]
        return objective
        
    def greedy_solution(self)-> tuple:
        """
        Greedy solution algorithm for the vehicle routing problem. Gives a good starting solution to the vehicle routing problem
        through assigning the nearest customer to each vehicle at each step until the vehicle is full.

        Returns
        -------
        solution : dict
            Dictionary containing the solution to the vehicle routing problem.
        objective : int
            Objective function value of the given solution.
        Capacity : dict
            Dictionary containing the capacity of each vehicle.
        """
        
        # Initialize solution, capacity and customer list
        to_be_served = list(range(1,self.customers+1))
        vehicle_count = list(range(1,self.vehicles+1))
        capacity = {}
        solution = {}
        for i in range(1, self.vehicles + 1):
            solution[i] = [0]
            capacity[i] = 0
        
        # Assign nearest customer to each vehicle if vehicle is not full
        for i in range (self.customers):
            for j in (vehicle_count):
                if to_be_served:
                    start_node = solution[j][-1]
                    next_node_index = np.argmin(self.distance_matrix[start_node][to_be_served])
                    next_node = to_be_served[next_node_index]
                    
                    if capacity[j] + self.customers_demand[next_node] <= self.vehicle_capacity:
                        capacity[j] += self.customers_demand[next_node]
                        solution[j].append(next_node)
                        to_be_served.remove(next_node)
                        
        # If there are unassigned customers, try to swap already assigned customers between vehicles and try to make room for them
        if to_be_served:
            for i in vehicle_count[:-1]:
                for j in vehicle_count[i:]:
                    for s1 in solution[i][1:]:
                        for s2 in solution[j][1:]:
                            capacity_i = capacity[i] - self.customers_demand[s1] + self.customers_demand[s2]
                            capacity_j = capacity[j] + self.customers_demand[s1] - self.customers_demand[s2]
                            
                            for t in to_be_served:
                                if capacity_i + self.customers_demand[t] <= self.vehicle_capacity and capacity_j <= self.vehicle_capacity :
                                    solution[i].append(t)
                                    to_be_served.remove(t)
                                    capacity_i += self.customers_demand[t]
                                    capacity[i], capacity[j] = capacity_i, capacity_j                                    
                                    solution[i][solution[i].index(s1)], solution[j][solution[j].index(s2)] = s2, s1
                                    s1 = s2
                                    
                                elif  capacity_j + self.customers_demand[t] <= self.vehicle_capacity and capacity_i <= self.vehicle_capacity:
                                    solution[j].append(t)
                                    to_be_served.remove(t)
                                    capacity_j += self.customers_demand[t]
                                    capacity[i], capacity[j] = capacity_i, capacity_j
                                    solution[i][solution[i].index(s1)], solution[j][solution[j].index(s2)] = s2, s1
                                    s1 = s2
                                    
        # Apend the depot to the end of each vehicle
        for i in solution:
            solution[i].append(0)
        print(solution)
        
        # For each vehicle, use a travelling salesman heuristic to improve the solution's objective function
        for i in solution:
            solution[i] = self.tsp_improvement(solution,i)
        
        objective = self.calc_objective(solution)
        self.best_solution = solution
        print(to_be_served)
        print(objective)
        print(solution)
        return solution, objective, capacity
    
    
    def construct_neighbourhood(self,solution: dict, capacity: dict)-> tuple:
        """
        Constructs the neighbourhood of the current solution by selecting two random vehicles and swapping the customers.

        Parameters
        ----------
        solution : dict
            Dictionary containing the solution to the vehicle routing problem.
        capacity : dict
            Dictionary containing the capacity of each vehicle.

        Returns
        -------
        solution : dict
            Dictionary containing the solution to the vehicle routing problem after the neighbourhood has been constructed.
        capacity : dict
            Dictionary containing the capacity of each vehicle after the neighbourhood has been constructed.
        """
        
        # a is a parameter that determines if a swap was succesfully made between two vehicles or between a single vehicle.
        a = 0
        while a == 0:
            self.random_vehicle_1, self.random_vehicle_2 = random.sample (range(1,self.vehicles+1),2)
            random_way = random.randint(0,1)
            r1 = random.choice(solution[self.random_vehicle_1][1:-1])
            r2 = random.choice(solution[self.random_vehicle_2][1:-1])
            
            # Determine if a swap is made between two vehicles or between a single vehicle
            # Random_way = 0 means that the swap is between two vehicles
            if random_way == 0:
                if (
                        capacity[self.random_vehicle_1] + self.customers_demand[r2] - self.customers_demand[r1] <= self.vehicle_capacity 
                    and capacity[self.random_vehicle_2] + self.customers_demand[r1] - self.customers_demand[r2] <= self.vehicle_capacity
                    ):
                    first_index = solution[self.random_vehicle_1].index(r1)
                    second_index = solution[self.random_vehicle_2].index(r2)
                    
                    solution[self.random_vehicle_1][first_index] = r2
                    solution[self.random_vehicle_2][second_index] = r1
                    capacity[self.random_vehicle_1] += self.customers_demand[r2] - self.customers_demand[r1]
                    capacity[self.random_vehicle_2] += self.customers_demand[r1] - self.customers_demand[r2]
                    a +=1
                    return solution, capacity
                
            # Random_way = 1 means that the swap is between a single vehicle   
            else:
                if capacity[self.random_vehicle_1] + self.customers_demand[r2] <= self.vehicle_capacity:
                    index = solution[self.random_vehicle_1].index(r1)
                    solution[self.random_vehicle_1].insert(index,r2)
                    solution[self.random_vehicle_2].remove(r2)
                    capacity[self.random_vehicle_1] += self.customers_demand[r2]
                    capacity[self.random_vehicle_2] -= self.customers_demand[r2]
                    a += 1
                    return solution, capacity
                
                elif capacity[self.random_vehicle_2] + self.customers_demand[r1] <= self.vehicle_capacity:
                    index = solution[self.random_vehicle_2].index(r2)
                    solution[self.random_vehicle_2].insert(index,r1)
                    solution[self.random_vehicle_1].remove(r1)
                    capacity[self.random_vehicle_2] += self.customers_demand[r1]
                    capacity[self.random_vehicle_1] -= self.customers_demand[r1]
                    a += 1 
                    return solution, capacity
                
    # Define the Metropolis criteria
    def metropolis_criteria(self,delta:float)-> float:
        return math.exp(-delta / self.temperature)
    
    # Define acceptance criteria          
    def accept_criteria(self):
        delta = self.cand_obj - self.cur_obj
        
        # If the new solution is better than the current solution, accept it
        if delta < 0:
            self.cur_solution = self.cand_solution
            self.cur_obj = self.cand_obj
            self.cur_capacity = self.cur_capacity
            if self.cur_obj < self.best_objective:
                self.update_best()
                
        # If the new solution is worse than the current solution, accept it if the metropolis criteria
        # is greater than a random number uniformly distributed between 0 and 1
        else:
            if random.random() < self.metropolis_criteria(delta):
                self.cur_solution = self.cand_solution
                self.cur_obj = self.cand_obj
                self.cur_capacity = self.cur_capacity
        pass            
    
    # Updates the best solution
    def update_best(self):
            self.best_solution = copy.deepcopy(self.cur_solution)
            self.best_objective = self.cur_obj
            
    def tsp_improvement(self, solution, i):
        """
        Use a travelling salesman heuristic on individual vehicles to improve the solution's objective function.

        Parameters
        ----------
        solution : dict
            Dictionary containing the solution to the vehicle routing problem.
        i : int
            Vehicle number.

        Returns
        -------
        best_solution : dict
            Currently best solution to the vehicle routing problem.
        """
        
        candidate_tsp = solution[i][::]
        best_solution = solution[i][::]
        improvement = True
        
        while improvement:
            improvement = False
            # Use a two-opt neighbourhood swap to improve the solution's objective function
            for i in range(1,len(candidate_tsp)-2):
                for j in range(i+1,len(candidate_tsp)-1):
                    first = candidate_tsp[i]
                    second = candidate_tsp[j]
                    difference = (self.distance_matrix[candidate_tsp[i-1]][candidate_tsp[i]] 
                                  + self.distance_matrix[candidate_tsp[i]][candidate_tsp[j]] 
                                  - self.distance_matrix[candidate_tsp[i-1]][candidate_tsp[j]] 
                                  - self.distance_matrix[candidate_tsp[j]][candidate_tsp[i]] )

                    if difference > 0:
                        candidate_tsp[i] = second
                        candidate_tsp[j] = first
                        best_solution = candidate_tsp[::]
                        improvement = True
        return best_solution
                        
                
                
    def annealing(self):
        """
        Define the simulated annealing algorithm.
        """
        # Get greedy solution
        self.cur_solution, self.cur_obj, self.cur_capacity = self.greedy_solution()
        self.update_best()
        
        # Loop until the temperature is less than a threshold, or iterations limit is reached
        while self.temperature >= self.stop_temperature and self.iterations != 0:
            self.iterations -= 1
            # Construct a candidate solution
            self.cand_solution, self.cand_capacity = self.construct_neighbourhood(self.cur_solution, self.cur_capacity)
            
            # Improve the candidate solution using a travelling salesman heuristic
            for r in [self.random_vehicle_1, self.random_vehicle_2]:
                self.cand_solution[r] = self.tsp_improvement(self.cand_solution,r)
            self.cand_obj = self.calc_objective(self.cand_solution)
            
            # Use the acceptance criteria to accept or reject the candidate solution and update the current solution
            self.accept_criteria()
            self.temperature *= self.cooling_rate
            
        print(self.best_solution)
        print(self.best_objective)
        print(self.temperature, "temperature", self.current_iteration - self.iterations,"iterations")
        

if __name__ == "__main__":
    args = "instance_10_3_1.txt"

    inst = Instance(args)
    print("Loaded instance: {0}".format(inst))
    start_time = time.time()
    sa =  Simulated_Annealing(inst,10000,1,10000, 0.99)
    end_time = time.time()
    duration = end_time - start_time
    result = sa.annealing()
    print("duration",duration)





