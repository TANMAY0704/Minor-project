import math

def dfs(node, parent, graph, costs, visited):
    visited[node] = True
    
    for child, weight in graph[node]:
        if not visited[child]:
            dfs(child, node, graph, costs, visited)
            
            # Calculate cost for each edge considering number of times traversed
            times_traversed = costs[child][0]
            edge_cost = weight * math.ceil(times_traversed / 2)
            costs[node][1] += edge_cost
            costs[node][0] += times_traversed
    
    if parent is not None:
        # Update cost for edge between parent and node
        times_traversed = costs[node][0]  # Number of times parent node has been visited
        weight = next(weight for next_node, weight in graph[parent] if next_node == node)
        edge_cost = weight * math.ceil(times_traversed / 2)
        costs[node][1] += edge_cost

def findminimumCost(tree_nodes, tree_from, tree_to, tree_weight, start, end):
    # Initialize the graph
    graph = {i: [] for i in range(1, tree_nodes + 1)}
    for i in range(len(tree_from)):
        u, v, w = tree_from[i], tree_to[i], tree_weight[i]
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Initialize costs array to track number of times each node is visited and total cost
    costs = [[0, 0] for _ in range(tree_nodes + 1)]  # [times visited, total cost]
    
    # Perform DFS to calculate costs and visits
    visited = [False] * (tree_nodes + 1)
    dfs(start, None, graph, costs, visited)
    
    # Calculate the minimum cost path from start to end
    min_cost = costs[end][1]
    
    return min_cost

# Sample Input
tree_nodes = 3
tree_from = [1, 2]
tree_to = [2, 3]
tree_weight = [10, 20]
start = 1
end = 3

# Output
print(findminimumCost(tree_nodes, tree_from, tree_to, tree_weight, start, end))
