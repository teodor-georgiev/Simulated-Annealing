## Simulated Annealing (SA) algorithm for the Vehicle Routing Problem (VRP)

The objective of the Vehicle Routing Problem (VRP) is to minimize the total distance traveled by a fleet of vehicles while satisfying the demands of a set of customers. The problem has the following constraints and information:

- A route must start and end at the depot.
- All vehicles must be utilized.
- All vehicles have the same capacity.
- Each customer can only be served by one vehicle.
- Customers can have varying demands.
- The order of the customers on a route is irrelevant.
- The objective is to minimize the total distance covered by all vehicles.

To solve this problem, we will use a metaheuristic called Simulated Annealing (SA), which is a probabilistic technique for approximating the global optimum of a given function.

## Greedy algorithm for the initial solution
To obtain a good initial solution before running the SA algorithm, we will use a greedy heuristic. The implemented greedy heuristic assigns the closest customer to a vehicle at each step until the vehicle's capacity is reached. If there are unassigned customers remaining, the algorithm attempts to reassign customers between the vehicles until there is available space for one.

## Simulated Annealing (SA)
Once we have obtained an initial solution from the greedy heuristic, we will start the Simulated Annealing (SA) metaheuristic. At each iteration of the algorithm, we will construct a candidate solution and objective value by generating a neighborhood solution. We will generate the neighborhood solution by either swapping random customers between two randomly chosen vehicles or by swapping customers within a single vehicle. After generating the candidate solution, we will perform a simple two-opt traveling salesman improvement heuristic to further improve its objective function.

If the candidate solution is better than the current solution, it will always be accepted. If it is worse, we will accept it with a probability determined by the Metropolis criteria, which is defined by:
<p align="center">
  <img src="metropolis criteria.PNG", width = 800 />
</p>
With:

- $s$ current solution
- $s^{′}$ candidate solution
- $T$ Temperature
