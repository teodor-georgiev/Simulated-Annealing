## Simulated Annealing (SA) algorithm for the Vehicle Routing Problem (VRP)

The objective of the vehicle routing problem (VRP) is to minimize the total distance
traveled by a fleet of vehicles that must visit a set of customers to satisfy their demands
for a product. In this case of the problem we have the following constraints and information:
- A route starts and ends at the depot
- All vehicles must be used
- All vehicles have the same capacity
- A customer can only be served by one vehicle
- Each customer has a varying demand
- The order of the customers on a route is not important
- The objective is to minimize the sum of the distance covered by all vehicles

To solve this problem, we will use a __metaheuristic__ known as Simulated Annealing (SA) which is a probabilistic technique for approximating the global optimum of a given function.

## Greedy algorithm for the initial solution
Before we start the SA algorithm we will first use a greedy heuristic to get a somewhat good initial solution. The greedy heuristic implemented here tries at each step to assign the closest customer to one vehicle until there is no more space left. If there are any unassigned customers left try to shuffle the customers between the vehicles until there is space for one.

## Simulated Annealing (SA)
After we have our initial solution given by the greedy heuristic, we will be starting the Simulated Annealing metaheuristic. At every iteration of the algorithm, we will be getting a candidate solution and objective value by constructing a neighborhood solution, either by swapping random customers between two random vehicles or by swapping customers between a single vehicle. If a candidate solution is better than the current one it is always accepted, else it is accepted with the probability function of the metropolis criteria which is defined by:
<p align="center">
  <img src="metropolis criteria.PNG", width = 800 />
</p>
