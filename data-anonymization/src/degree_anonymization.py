"""
Problem 2: Degree-Anonymization for Graph Data

Step 1: Generate random graphs
Step 2: Obtain sorted degree sequences
Step 3: Dynamic programming for minimum-cost k-anonymization
Step 4: Check reliability using the Erdos-Gallai graphicality test
"""

import random


# ------------------------------------------------------------
# Edge class
# ------------------------------------------------------------
class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"({self.u}, {self.v})"


# ------------------------------------------------------------
# Result of DP anonymization
# ------------------------------------------------------------
class DPResult:
    def __init__(self, original, anonymized, cost):
        self.original = original
        self.anonymized = anonymized
        self.cost = cost


# ------------------------------------------------------------
# Step 1: Generate a random simple graph
# ------------------------------------------------------------
def generate_random_graph(n, m, rng):
    edges = []
    existing_edges = set()

    max_edges = n * (n - 1) // 2
    if m > max_edges:
        raise ValueError("Too many edges for a simple graph.")

    while len(edges) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)

        # No self-loops
        if u == v:
            continue

        # Store smaller vertex first
        a, b = min(u, v), max(u, v)
        key = (a, b)

        # Avoid duplicate edges
        if key not in existing_edges:
            existing_edges.add(key)
            edges.append(Edge(a, b))

    return edges


# ------------------------------------------------------------
# Step 2: Obtain degree sequence
# ------------------------------------------------------------
def get_degree_sequence(n, edges):
    degrees = [0] * n

    for edge in edges:
        degrees[edge.u] += 1
        degrees[edge.v] += 1

    # Sort descending
    degrees.sort(reverse=True)

    return degrees


# ------------------------------------------------------------
# Step 3: Dynamic Programming
#
# We only allow:
#
#       dHat[i] >= d[i]
#
# For a consecutive group d[j ... i-1],
# all degrees are raised to d[j].
#
# Group cost:
#
#       sum(d[j] - d[t])
#
# DP recurrence:
#
# DP[i] = min DP[j] + groupCost(j, i)
#
# where i - j >= k
# ------------------------------------------------------------
def degree_anonymization(input_degrees, k):
    d = sorted(input_degrees, reverse=True)
    n = len(d)

    INF = float("inf")
    dp = [INF] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = 0

    # --------------------------------------------------------
    # Dynamic Programming
    # --------------------------------------------------------
    for i in range(1, n + 1):
        # j is the beginning of the final group.
        # The final group must contain at least k elements:
        #       i - j >= k   =>   j <= i - k
        for j in range(0, i - k + 1):
            # Cost of making d[j ... i-1] equal to d[j]
            group_cost = 0
            for t in range(j, i):
                group_cost += d[j] - d[t]

            candidate = dp[j] + group_cost

            if candidate < dp[i]:
                dp[i] = candidate
                parent[i] = j

    # --------------------------------------------------------
    # Reconstruct optimal groups
    # --------------------------------------------------------
    groups = []
    current = n

    while current > 0:
        start = parent[current]

        if start == -1:
            raise RuntimeError("No valid k-anonymous solution found.")

        groups.append((start, current))
        current = start

    groups.reverse()

    # --------------------------------------------------------
    # Construct d_hat
    # --------------------------------------------------------
    d_hat = [0] * n

    for start, end in groups:
        generalized_value = d[start]
        for i in range(start, end):
            d_hat[i] = generalized_value

    return DPResult(d, d_hat, int(dp[n]))


# ------------------------------------------------------------
# Check k-anonymity
#
# Every degree value must occur at least k times.
# ------------------------------------------------------------
def is_k_anonymous(degree_sequence, k):
    frequency = {}

    for degree in degree_sequence:
        frequency[degree] = frequency.get(degree, 0) + 1

    for count in frequency.values():
        if count < k:
            return False

    return True


# ------------------------------------------------------------
# Step 4: Erdos-Gallai Graphicality Test
#
# Determines whether a degree sequence can represent
# a simple undirected graph.
# ------------------------------------------------------------
def is_graphical(input_seq):
    d = sorted(input_seq, reverse=True)
    n = len(d)

    # Basic conditions
    for degree in d:
        if degree < 0 or degree >= n:
            return False

    # Sum of degrees must be even
    if sum(d) % 2 != 0:
        return False

    # --------------------------------------------------------
    # Erdos-Gallai theorem
    # --------------------------------------------------------
    for k in range(1, n + 1):
        left = sum(d[i] for i in range(k))

        right = k * (k - 1)
        for i in range(k, n):
            right += min(d[i], k)

        if left > right:
            return False

    return True


# ------------------------------------------------------------
# Print helpers
# ------------------------------------------------------------
def print_sequence(sequence):
    print(sequence)


def print_edges(edges):
    print("Edges:")
    print(" ".join(str(e) for e in edges))


# ------------------------------------------------------------
# Run one experiment
# ------------------------------------------------------------
def run_experiment(n, m, k_values, rng):
    print()
    print("=" * 60)
    print(f"Graph: n = {n}, m = {m}")
    print("=" * 60)

    # Step 1
    edges = generate_random_graph(n, m, rng)
    print_edges(edges)

    # Step 2
    degree_sequence = get_degree_sequence(n, edges)
    print()
    print("Original degree sequence:")
    print_sequence(degree_sequence)

    # Step 3
    for k in k_values:
        print()
        print("-" * 60)
        print(f"k = {k}")

        result = degree_anonymization(degree_sequence, k)

        print("Anonymized degree sequence:")
        print_sequence(result.anonymized)

        print(f"Minimum anonymization cost: {result.cost}")
        print(f"k-anonymous: {is_k_anonymous(result.anonymized, k)}")

        # Step 4
        print(f"Graphical degree sequence: {is_graphical(result.anonymized)}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Fixed seed makes the experiments reproducible.
    # Note: Python's random generator differs from Java's,
    # so exact graph/degree sequences won't necessarily match
    # earlier Java/NetworkX results.
    rng = random.Random(2)

    # n = 6, m = 1.5 * 6 = 9, k = 2, 3
    run_experiment(6, 9, [2, 3], rng)

    # n = 10, m = 1.5 * 10 = 15, k = 3, 4
    run_experiment(10, 15, [3, 4], rng)

    # n = 20, m = 1.5 * 20 = 30, k = 3, 4
    run_experiment(20, 30, [3, 4], rng)


if __name__ == "__main__":
    main()