## Simulated Annealing SA algorithm for the Vehicle Routing Problem (VRP)

The objective of the vehicle routing problem (VRP) is to minimize the total distance
traveled of a fleet of vehicles that must visit a set of customers to satisfy their demands
for a product. In this case of the problem we have the following constraints and information:
- A route starts and ends at the depot
- All vehicles must be used
- All vehicles have the same capacity
- A customer can only be served by one vehicle
- Each customer has a varying demand
- The order of the customers on a route is not important
- The objective is to minimize the sum of the distance covered by all vehicles

In order to solve this problem we will be using a __metaheuristic__ known as Simulated Annealing (SA) which is a probabilistic technique for approximating the global optimum of a given function.

## Greedy algorihm for the intial solution
Before we start the SA algorithm we will first use a greedy heuristic to get a somewhat good iniial solution. The greedy heuristic implemented here tries at each step to assign the closest customer to one vehicle until there is no more space left. If there are any unassigned customers left 
