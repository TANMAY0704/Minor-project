import heapq
import math

def ceil(x):
    return math.ceil(x)

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if visited[node]:
            continue
        visited[node] = True
        for neighbor, weight in graph[node]:
            if not visited[neighbor] and d + weight < dist[neighbor]:
                dist[neighbor] = d + weight
                heapq.heappush(pq, (dist[neighbor], neighbor))
    return dist

def findminimumCost(tree_nodes, tree_from, tree_to, tree_weight, start, end):
    graph = [[] for _ in range(tree_nodes)]
    for i in range(len(tree_from)):
        graph[tree_from[i] - 1].append((tree_to[i] - 1, tree_weight[i]))
        graph[tree_to[i] - 1].append((tree_from[i] - 1, tree_weight[i]))

    dist = [dijkstra(graph, i) for i in range(tree_nodes)]

    dp = [[float('inf')] * tree_nodes for _ in range(1 << tree_nodes)]
    dp[1 << start][start] = 0

    for mask in range(1 << tree_nodes):
        for u in range(tree_nodes):
            if not (mask & (1 << u)):
                continue
            for v in range(tree_nodes):
                if mask & (1 << v):
                    continue
                for t in range(1, len(dist[u])):
                    cost = ceil(dist[u][v] * (t + 1) / 2)  # Corrected cost calculation
                    dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + cost)

    min_cost = float('inf')
    for i in range(tree_nodes):
        min_cost = min(min_cost, dp[(1 << tree_nodes) - 1][i] + ceil(dist[i][end] / 2))
    
    return min_cost

# Sample Input
tree_nodes = 3
tree_from = [1, 2]
tree_to = [2, 3]
tree_weight = [10, 20]
start = 1
end = 3

# Sample Output
print(findminimumCost(tree_nodes, tree_from, tree_to, tree_weight, start - 1, end - 1))  