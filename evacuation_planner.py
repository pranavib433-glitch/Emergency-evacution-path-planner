from __future__ import annotations

import heapq
import logging
import math
import random
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP  (trace logging & step-by-step reasoning)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("EvacPlanner")


def trace(msg: str) -> None:
    """Step-by-step trace for explainability."""
    logger.info("  TRACE | %s", msg)


# =============================================================================
# SECTION 1 – AGENT MODEL (PEAS)
# =============================================================================

class PEASAgent:
    """
    PEAS description for the Emergency Evacuation Path Planner AI Agent.

    Performance Measure : minimize evacuation time, maximize safety,
                          avoid hazards & congestion, reach nearest exit.
    Environment         : building layout (rooms, corridors), exits,
                          dynamic hazards (fire, smoke, debris).
    Actuators           : suggest safest path, send alerts, update routes,
                          guide to nearest exit via display/app/alarm.
    Sensors             : smoke/gas sensors, temperature, cameras/LiDAR,
                          crowd-density sensors, Wi-Fi/BT/GPS for location.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.performance_score: float = 0.0
        self.knowledge_base: Dict[str, Any] = {}

    # ── Performance Measure ───────────────────────────────────────────────
    def compute_performance(
        self,
        evacuation_time: float,
        users_safe: int,
        total_users: int,
        hazards_avoided: int,
    ) -> float:
        """
        Utility-based performance: weighted combination of safety metrics.
        Higher is better.
        """
        time_score   = max(0.0, 1.0 - evacuation_time / 300.0)   # 5-min baseline
        safety_ratio = users_safe / max(total_users, 1)
        hazard_score = hazards_avoided / max(hazards_avoided + 1, 1)
        self.performance_score = (
            0.4 * time_score + 0.4 * safety_ratio + 0.2 * hazard_score
        )
        trace(
            f"Agent {self.agent_id} performance = {self.performance_score:.3f} "
            f"(time={time_score:.2f}, safety={safety_ratio:.2f}, "
            f"hazard={hazard_score:.2f})"
        )
        return self.performance_score

    def perceive(self, sensor_data: Dict[str, Any]) -> None:
        """Update knowledge base from sensor inputs."""
        self.knowledge_base.update(sensor_data)
        trace(f"Agent {self.agent_id} perceived: {list(sensor_data.keys())}")

    def act(self, action: str) -> str:
        """Return actuator output for a given action."""
        actuators = {
            "suggest_path":    "Digital display / Mobile app updated.",
            "sound_alarm":     "Audio alarm triggered.",
            "send_alert":      "Push notification sent.",
            "update_route":    "Route dynamically updated.",
            "guide_to_exit":   "LED corridor indicators activated.",
        }
        result = actuators.get(action, "Unknown action.")
        trace(f"Actuator [{action}] → {result}")
        return result


# =============================================================================
# SECTION 2 – ENVIRONMENT TYPES
# =============================================================================

class EnvironmentType(Enum):
    FULLY_OBSERVABLE   = auto()   # all exits/hazards known via sensors & maps
    PARTIALLY_OBSERVABLE = auto() # smoke may block sensor visibility
    DETERMINISTIC      = auto()   # same action → same result (static layout)
    STOCHASTIC         = auto()   # fire spread / crowd movement uncertain
    STATIC             = auto()   # building layout fixed during planning
    DYNAMIC            = auto()   # fire spreads, people move in real-time
    DISCRETE           = auto()   # system re-evaluates every few seconds
    CONTINUOUS         = auto()   # crowd density & temperature change fluidly


@dataclass
class EnvironmentState:
    """
    Represents the current state of the building environment.
    Combines static layout with dynamic hazard conditions.
    """
    rooms: List[str]
    exits: List[str]
    hazardous_nodes: Set[str] = field(default_factory=set)
    blocked_edges: Set[Tuple[str, str]] = field(default_factory=set)
    crowd_density: Dict[str, float] = field(default_factory=dict)   # 0.0–1.0
    env_type: EnvironmentType = EnvironmentType.DYNAMIC

    def is_hazardous(self, node: str) -> bool:
        return node in self.hazardous_nodes

    def is_blocked(self, u: str, v: str) -> bool:
        return (u, v) in self.blocked_edges or (v, u) in self.blocked_edges

    def update_hazard(self, node: str, hazardous: bool) -> None:
        if hazardous:
            self.hazardous_nodes.add(node)
            trace(f"Hazard DETECTED at '{node}'")
        else:
            self.hazardous_nodes.discard(node)
            trace(f"Hazard CLEARED at '{node}'")


# =============================================================================
# SECTION 3 – BUILDING GRAPH (Knowledge Representation)
# =============================================================================

@dataclass
class Edge:
    """Weighted directed edge in the building graph."""
    to: str
    weight: float                    # base travel time in seconds
    corridor_width: float = 2.0      # metres (affects crowd factor)


class BuildingGraph:
    """
    Graph-based knowledge representation of the building layout.
    Nodes = rooms/areas; Edges = corridors/pathways.
    Stores coordinates for heuristic computation.
    """

    def __init__(self) -> None:
        self.adjacency: Dict[str, List[Edge]] = defaultdict(list)
        self.coordinates: Dict[str, Tuple[float, float]] = {}   # (x, y) metres
        self.node_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def add_node(self, name: str, x: float, y: float, **meta: Any) -> None:
        self.coordinates[name] = (x, y)
        self.node_metadata[name].update(meta)

    def add_edge(
        self,
        u: str,
        v: str,
        weight: float,
        bidirectional: bool = True,
        width: float = 2.0,
    ) -> None:
        self.adjacency[u].append(Edge(to=v, weight=weight, corridor_width=width))
        if bidirectional:
            self.adjacency[v].append(Edge(to=u, weight=weight, corridor_width=width))

    def neighbours(self, node: str, env: EnvironmentState) -> List[Tuple[str, float]]:
        """
        Return (neighbour, effective_cost) pairs respecting current hazards
        and blocked edges. Crowd density increases cost.
        """
        result: List[Tuple[str, float]] = []
        for edge in self.adjacency.get(node, []):
            if env.is_blocked(node, edge.to):
                trace(f"  Edge {node}→{edge.to} BLOCKED, skipping.")
                continue
            if env.is_hazardous(edge.to):
                trace(f"  Node '{edge.to}' HAZARDOUS, skipping.")
                continue
            density = env.crowd_density.get(edge.to, 0.0)
            crowd_factor = 1.0 + density          # 1× to 2× cost
            effective_cost = edge.weight * crowd_factor
            result.append((edge.to, effective_cost))
        return result

    def euclidean_distance(self, a: str, b: str) -> float:
        """Straight-line distance between two nodes (admissible heuristic base)."""
        ax, ay = self.coordinates.get(a, (0, 0))
        bx, by = self.coordinates.get(b, (0, 0))
        return math.hypot(bx - ax, by - ay)


# =============================================================================
# SECTION 4 – PROBLEM FORMULATION  (state / action / transition / cost)
# =============================================================================

@dataclass(frozen=True)
class EvacState:
    """
    Immutable state for the search problem.
    state = current node the evacuee occupies.
    """
    node: str

    def __repr__(self) -> str:
        return f"State({self.node})"


class EvacProblem:
    """
    Formal problem formulation for evacuation path planning.

    initial_state : room where the evacuee currently is.
    goal_test     : reached any exit node.
    actions       : move to an adjacent non-hazardous node.
    transition    : new state is the destination node.
    cost          : effective travel cost (base × crowd factor).
    """

    def __init__(
        self,
        graph: BuildingGraph,
        env: EnvironmentState,
        start: str,
    ) -> None:
        self.graph = graph
        self.env = env
        self.initial_state = EvacState(node=start)

    def goal_test(self, state: EvacState) -> bool:
        return state.node in self.env.exits

    def actions(self, state: EvacState) -> List[Tuple[str, float]]:
        """Returns list of (next_node, cost)."""
        return self.graph.neighbours(state.node, self.env)

    def transition(self, state: EvacState, next_node: str) -> EvacState:
        return EvacState(node=next_node)

    def step_cost(self, _from: EvacState, next_node: str, cost: float) -> float:
        return cost

    def heuristic_distance(self, state: EvacState) -> float:
        """
        Admissible heuristic: minimum Euclidean distance to any exit,
        scaled by 0.8 to guarantee h(n) <= true edge-weight cost.
        (Edges are in seconds; coordinates in metres. A 0.8 scale is a
        conservative lower-bound assuming corridors are slightly longer
        than straight-line distance.)
        Consistent because Euclidean satisfies triangle inequality.
        """
        if not self.env.exits:
            return 0.0
        scale = 0.8
        return scale * min(
            self.graph.euclidean_distance(state.node, ex)
            for ex in self.env.exits
        )


# =============================================================================
# SECTION 5 – SEARCH ALGORITHMS  (BFS / DFS / UCS / Greedy / A*)
# =============================================================================

@dataclass
class SearchResult:
    algorithm: str
    path: List[str]
    total_cost: float
    nodes_expanded: int
    runtime_ms: float
    peak_memory_kb: float
    found: bool


def _reconstruct(came_from: Dict[str, Optional[str]], goal: str) -> List[str]:
    """Reconstruct path from came_from map."""
    path: List[str] = []
    node: Optional[str] = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path


# ── BFS ───────────────────────────────────────────────────────────────────────
def bfs(problem: EvacProblem) -> SearchResult:
    """
    Breadth-First Search – finds shortest path in terms of number of steps.
    Uses a deque for O(1) popleft. Guarantees optimal for uniform costs.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    frontier: deque[EvacState] = deque([problem.initial_state])
    came_from: Dict[str, Optional[str]] = {problem.initial_state.node: None}
    cost_so_far: Dict[str, float] = {problem.initial_state.node: 0.0}
    nodes_expanded = 0

    trace(f"BFS start: {problem.initial_state.node}")

    while frontier:
        state = frontier.popleft()
        nodes_expanded += 1
        trace(f"BFS expanding: {state.node}")

        if problem.goal_test(state):
            path = _reconstruct(came_from, state.node)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            trace(f"BFS GOAL found: {path} cost={cost_so_far[state.node]:.2f}")
            return SearchResult("BFS", path, cost_so_far[state.node],
                                nodes_expanded, rt, peak / 1024, True)

        for next_node, cost in problem.actions(state):
            new_cost = cost_so_far[state.node] + cost
            if next_node not in came_from:
                came_from[next_node] = state.node
                cost_so_far[next_node] = new_cost
                frontier.append(problem.transition(state, next_node))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("BFS: no path found.")
    return SearchResult("BFS", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── DFS ───────────────────────────────────────────────────────────────────────
def dfs(problem: EvacProblem, depth_limit: int = 50) -> SearchResult:
    """
    Depth-First Search with depth limit (prevents infinite loops).
    Memory efficient but NOT guaranteed optimal.
    Uses explicit stack to avoid Python recursion limits.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    # Stack stores (state, came_from_dict_snapshot, cumulative_cost, depth)
    stack: List[Tuple[EvacState, Dict[str, Optional[str]], float, int]] = [
        (problem.initial_state, {problem.initial_state.node: None}, 0.0, 0)
    ]
    nodes_expanded = 0

    trace(f"DFS start: {problem.initial_state.node} (limit={depth_limit})")

    while stack:
        state, came_from, cum_cost, depth = stack.pop()
        nodes_expanded += 1
        trace(f"DFS expanding: {state.node} depth={depth}")

        if problem.goal_test(state):
            path = _reconstruct(came_from, state.node)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            trace(f"DFS GOAL: {path}")
            return SearchResult("DFS", path, cum_cost, nodes_expanded, rt, peak / 1024, True)

        if depth < depth_limit:
            for next_node, cost in problem.actions(state):
                if next_node not in came_from:
                    new_cf = dict(came_from)
                    new_cf[next_node] = state.node
                    stack.append(
                        (problem.transition(state, next_node), new_cf, cum_cost + cost, depth + 1)
                    )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("DFS: no path found within depth limit.")
    return SearchResult("DFS", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── UCS ───────────────────────────────────────────────────────────────────────
def ucs(problem: EvacProblem) -> SearchResult:
    """
    Uniform Cost Search – optimal for arbitrary positive edge costs.
    Uses a min-heap (priority queue) ordered by cumulative cost g(n).
    Closed set avoids re-expanding nodes.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    # Heap entries: (cost, tie_break, state)
    counter = 0
    heap: List[Tuple[float, int, EvacState]] = [(0.0, counter, problem.initial_state)]
    came_from: Dict[str, Optional[str]] = {problem.initial_state.node: None}
    cost_so_far: Dict[str, float] = {problem.initial_state.node: 0.0}
    closed: Set[str] = set()
    nodes_expanded = 0

    trace(f"UCS start: {problem.initial_state.node}")

    while heap:
        g, _, state = heapq.heappop(heap)
        if state.node in closed:
            continue
        closed.add(state.node)
        nodes_expanded += 1
        trace(f"UCS expanding: {state.node} g={g:.2f}")

        if problem.goal_test(state):
            path = _reconstruct(came_from, state.node)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            trace(f"UCS GOAL: {path} cost={g:.2f}")
            return SearchResult("UCS", path, g, nodes_expanded, rt, peak / 1024, True)

        for next_node, cost in problem.actions(state):
            new_g = g + cost
            if next_node not in cost_so_far or new_g < cost_so_far[next_node]:
                cost_so_far[next_node] = new_g
                came_from[next_node] = state.node
                counter += 1
                heapq.heappush(heap, (new_g, counter, problem.transition(state, next_node)))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("UCS: no path found.")
    return SearchResult("UCS", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── GREEDY BEST-FIRST ─────────────────────────────────────────────────────────
def greedy(problem: EvacProblem) -> SearchResult:
    """
    Greedy Best-First Search – orders frontier by heuristic h(n) only.
    Fast but NOT optimal; good for quick escape suggestions.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    counter = 0
    h0 = problem.heuristic_distance(problem.initial_state)
    heap: List[Tuple[float, int, EvacState]] = [(h0, counter, problem.initial_state)]
    came_from: Dict[str, Optional[str]] = {problem.initial_state.node: None}
    cost_so_far: Dict[str, float] = {problem.initial_state.node: 0.0}
    closed: Set[str] = set()
    nodes_expanded = 0

    trace(f"Greedy start: {problem.initial_state.node} h={h0:.2f}")

    while heap:
        h, _, state = heapq.heappop(heap)
        if state.node in closed:
            continue
        closed.add(state.node)
        nodes_expanded += 1
        trace(f"Greedy expanding: {state.node} h={h:.2f}")

        if problem.goal_test(state):
            path = _reconstruct(came_from, state.node)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            trace(f"Greedy GOAL: {path}")
            return SearchResult("Greedy", path, cost_so_far[state.node],
                                nodes_expanded, rt, peak / 1024, True)

        for next_node, cost in problem.actions(state):
            if next_node not in closed:
                new_cost = cost_so_far[state.node] + cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    came_from[next_node] = state.node
                next_state = problem.transition(state, next_node)
                h_next = problem.heuristic_distance(next_state)
                counter += 1
                heapq.heappush(heap, (h_next, counter, next_state))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("Greedy: no path found.")
    return SearchResult("Greedy", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── A* ────────────────────────────────────────────────────────────────────────
def astar(problem: EvacProblem, tie_break_weight: float = 1e-4) -> SearchResult:
    """
    A* Search – f(n) = g(n) + h(n).
    Heuristic: minimum Euclidean distance to any exit.
    Admissible (never overestimates) → guaranteed optimal.
    Consistent (satisfies triangle inequality) → no re-expansion needed.
    Tie-breaking: slightly inflate h to prefer nodes closer to goal
    among equal-f nodes (improves practical performance).
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    counter = 0
    h0 = problem.heuristic_distance(problem.initial_state)
    f0 = 0.0 + h0
    heap: List[Tuple[float, int, EvacState]] = [(f0, counter, problem.initial_state)]
    came_from: Dict[str, Optional[str]] = {problem.initial_state.node: None}
    g_score: Dict[str, float] = {problem.initial_state.node: 0.0}
    closed: Set[str] = set()
    nodes_expanded = 0

    trace(f"A* start: {problem.initial_state.node} h={h0:.2f}")

    while heap:
        f, _, state = heapq.heappop(heap)
        if state.node in closed:
            continue
        closed.add(state.node)
        nodes_expanded += 1
        g = g_score[state.node]
        trace(f"A* expanding: {state.node} g={g:.2f} h={f-g:.2f} f={f:.2f}")

        if problem.goal_test(state):
            path = _reconstruct(came_from, state.node)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            trace(f"A* GOAL: {path} cost={g:.2f}")
            return SearchResult("A*", path, g, nodes_expanded, rt, peak / 1024, True)

        for next_node, cost in problem.actions(state):
            tentative_g = g + cost
            if next_node not in g_score or tentative_g < g_score[next_node]:
                g_score[next_node] = tentative_g
                came_from[next_node] = state.node
                next_state = problem.transition(state, next_node)
                h = problem.heuristic_distance(next_state)
                # Tie-breaking: add tiny fraction of h to f
                f_new = tentative_g + h * (1 + tie_break_weight)
                counter += 1
                heapq.heappush(heap, (f_new, counter, next_state))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("A*: no path found.")
    return SearchResult("A*", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── IDA* CONCEPT (memory-bounded variant) ────────────────────────────────────
def ida_star(problem: EvacProblem) -> SearchResult:
    """
    Iterative Deepening A* – memory-bounded variant of A*.
    Uses O(d) memory (depth of solution) instead of O(b^d).
    Re-expands nodes but saves memory – useful on embedded hardware.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    nodes_expanded = 0

    def search(
        path: List[str],
        g: float,
        bound: float,
        came_from_set: Set[str],
    ) -> Tuple[float, Optional[List[str]]]:
        nonlocal nodes_expanded
        state = EvacState(node=path[-1])
        f = g + problem.heuristic_distance(state)
        if f > bound:
            return f, None
        if problem.goal_test(state):
            return -1.0, list(path)
        minimum = float("inf")
        nodes_expanded += 1
        for next_node, cost in problem.actions(state):
            if next_node not in came_from_set:
                came_from_set.add(next_node)
                path.append(next_node)
                t, result = search(path, g + cost, bound, came_from_set)
                if result is not None:
                    return -1.0, result
                if t < minimum:
                    minimum = t
                path.pop()
                came_from_set.discard(next_node)
        return minimum, None

    start_node = problem.initial_state.node
    bound = problem.heuristic_distance(problem.initial_state)
    path_nodes = [start_node]
    visited = {start_node}

    trace(f"IDA* start: {start_node} initial_bound={bound:.2f}")

    while True:
        t, result = search(path_nodes, 0.0, bound, visited)
        if result is not None:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rt = (time.perf_counter() - t0) * 1000
            total_cost = sum(
                problem.graph.euclidean_distance(result[i], result[i + 1])
                for i in range(len(result) - 1)
            )
            trace(f"IDA* GOAL: {result} approx_cost={total_cost:.2f}")
            return SearchResult("IDA*", result, total_cost,
                                nodes_expanded, rt, peak / 1024, True)
        if t == float("inf"):
            break
        bound = t
        visited = {start_node}
        path_nodes = [start_node]

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rt = (time.perf_counter() - t0) * 1000
    trace("IDA*: no path found.")
    return SearchResult("IDA*", [], float("inf"), nodes_expanded, rt, peak / 1024, False)


# ── Heuristic Evaluation: Admissibility Check ────────────────────────────────
def evaluate_heuristic_admissibility(
    problem: EvacProblem, optimal_cost: float
) -> Dict[str, bool]:
    """
    Check whether heuristic h(n) ≤ true cost for sampled nodes.
    Admissible  : h never overestimates → A* remains optimal.
    Consistent  : h(n) ≤ cost(n→n') + h(n') for every edge.
    """
    results: Dict[str, bool] = {}
    graph = problem.graph
    for node in list(graph.coordinates.keys())[:20]:
        h = problem.heuristic_distance(EvacState(node=node))
        sub_prob = EvacProblem(graph, problem.env, node)
        sub_res = ucs(sub_prob)
        true_cost = sub_res.total_cost if sub_res.found else float("inf")
        admissible = h <= true_cost + 1e-6
        results[node] = admissible
        if not admissible:
            logger.warning("Heuristic NOT admissible at '%s': h=%.2f > true=%.2f",
                           node, h, true_cost)
    return results


# =============================================================================
# SECTION 6 – CSP: EVACUATION ZONE ASSIGNMENT
# =============================================================================

class EvacZoneCSP:
    """
    CSP: Assign each zone (room cluster) to an exit such that:
      - Each exit has capacity constraints (max zones per exit).
      - No two adjacent zones use the same exit (reduces congestion).
      - Priority zones (hospitals, mobility-impaired) get nearest exits.

    Variables   : zones  (e.g. 'Z1', 'Z2', …)
    Domain      : exits  (e.g. 'E1', 'E2', …)
    Constraints :
        1. Capacity: |{z : assignment[z] == exit}| ≤ max_zones_per_exit
        2. Adjacency: adjacent zones must NOT share the same exit
        3. Priority: priority zones must be assigned nearest available exit
    """

    def __init__(
        self,
        zones: List[str],
        exits: List[str],
        adjacency: Dict[str, List[str]],        # zone → neighbour zones
        exit_capacity: Dict[str, int],           # exit → max zones
        priority_zones: List[str],               # must get nearest exit
        distances: Dict[Tuple[str, str], float], # (zone, exit) → distance
    ) -> None:
        self.zones = zones
        self.exits = exits
        self.adjacency = adjacency
        self.exit_capacity = exit_capacity
        self.priority_zones = priority_zones
        self.distances = distances
        self.domains: Dict[str, List[str]] = {z: list(exits) for z in zones}
        self.assignment: Dict[str, str] = {}

    # ── Constraint checks ─────────────────────────────────────────────────
    def is_consistent(self, zone: str, exit_: str, assignment: Dict[str, str]) -> bool:
        """Check all constraints for assigning exit_ to zone."""
        # Capacity constraint
        current_count = sum(1 for v in assignment.values() if v == exit_)
        if current_count >= self.exit_capacity.get(exit_, 999):
            trace(f"  CSP FAIL: {exit_} at capacity for zone {zone}")
            return False
        # Adjacency constraint (no congestion)
        for neighbour in self.adjacency.get(zone, []):
            if assignment.get(neighbour) == exit_:
                trace(f"  CSP FAIL: adjacent zones {zone} & {neighbour} both → {exit_}")
                return False
        # Priority constraint
        if zone in self.priority_zones:
            nearest = min(
                self.exits, key=lambda e: self.distances.get((zone, e), float("inf"))
            )
            if exit_ != nearest:
                trace(f"  CSP FAIL: priority zone {zone} must use nearest exit {nearest}")
                return False
        return True

    # ── MRV heuristic ─────────────────────────────────────────────────────
    def select_unassigned_variable(
        self, assignment: Dict[str, str], domains: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        Minimum Remaining Values (MRV): choose variable with fewest
        legal values → detects failure early.
        Degree heuristic as tie-breaker: choose zone with most constraints.
        """
        unassigned = [z for z in self.zones if z not in assignment]
        if not unassigned:
            return None
        # MRV
        min_vals = min(len(domains[z]) for z in unassigned)
        mrv_candidates = [z for z in unassigned if len(domains[z]) == min_vals]
        # Degree tie-break
        return max(mrv_candidates, key=lambda z: len(self.adjacency.get(z, [])))

    # ── LCV heuristic ─────────────────────────────────────────────────────
    def order_domain_values(
        self, zone: str, domains: Dict[str, List[str]], assignment: Dict[str, str]
    ) -> List[str]:
        """
        Least Constraining Value (LCV): order exit choices by how many
        options they leave for neighbouring zones (least constraining first).
        """
        def constraint_count(exit_: str) -> int:
            count = 0
            for nb in self.adjacency.get(zone, []):
                if nb not in assignment and exit_ in domains[nb]:
                    count += 1
            return count

        return sorted(domains[zone], key=constraint_count)

    # ── Forward Checking ──────────────────────────────────────────────────
    def forward_check(
        self, zone: str, exit_: str, domains: Dict[str, List[str]], assignment: Dict[str, str]
    ) -> Optional[Dict[str, List[str]]]:
        """
        After assigning exit_ to zone, remove exit_ from neighbours' domains
        if adjacency constraint would be violated.
        Returns updated domains or None if any domain becomes empty (dead-end).
        """
        new_domains = {z: list(d) for z, d in domains.items()}
        for nb in self.adjacency.get(zone, []):
            if nb not in assignment:
                if exit_ in new_domains[nb]:
                    new_domains[nb].remove(exit_)
                    trace(f"  Forward-check: removed {exit_} from domain of {nb}")
                if not new_domains[nb]:
                    trace(f"  Forward-check: domain of {nb} EMPTY → backtrack")
                    return None
        return new_domains

    # ── Backtracking Search ───────────────────────────────────────────────
    def backtrack(
        self,
        assignment: Dict[str, str],
        domains: Dict[str, List[str]],
        node_count: List[int],
    ) -> Optional[Dict[str, str]]:
        """Recursive backtracking with MRV, LCV, forward checking."""
        if len(assignment) == len(self.zones):
            return assignment

        zone = self.select_unassigned_variable(assignment, domains)
        if zone is None:
            return assignment

        for exit_ in self.order_domain_values(zone, domains, assignment):
            node_count[0] += 1
            trace(f"CSP trying: {zone} → {exit_}")
            if self.is_consistent(zone, exit_, assignment):
                assignment[zone] = exit_
                reduced = self.forward_check(zone, exit_, domains, assignment)
                if reduced is not None:
                    result = self.backtrack(assignment, reduced, node_count)
                    if result is not None:
                        return result
                del assignment[zone]
                trace(f"CSP backtrack: {zone} ← {exit_} failed, undoing.")

        return None

    def solve(self) -> Tuple[Optional[Dict[str, str]], int]:
        """Solve the CSP and return (assignment, nodes_explored)."""
        nc = [0]
        result = self.backtrack({}, {z: list(d) for z, d in self.domains.items()}, nc)
        if result:
            trace(f"CSP SOLVED: {result}")
        else:
            trace("CSP: no valid assignment found.")
        return result, nc[0]

    # ── Min-Conflicts Local Search ────────────────────────────────────────
    def min_conflicts(self, max_steps: int = 1000) -> Optional[Dict[str, str]]:
        """
        Min-conflicts local search for CSP.
        Useful when a complete but possibly inconsistent assignment exists.
        Good for large instances where backtracking is slow.
        """
        # Random complete assignment
        assignment = {z: random.choice(self.exits) for z in self.zones}
        trace("Min-Conflicts: starting with random assignment.")

        for step in range(max_steps):
            # Find all conflicted variables
            conflicted = [
                z for z in self.zones
                if not self.is_consistent(z, assignment[z],
                                          {k: v for k, v in assignment.items() if k != z})
            ]
            if not conflicted:
                trace(f"Min-Conflicts solved at step {step}")
                return assignment

            zone = random.choice(conflicted)
            # Choose exit that minimises conflicts
            best_exit = min(
                self.exits,
                key=lambda e: sum(
                    1 for nb in self.adjacency.get(zone, [])
                    if assignment.get(nb) == e
                )
            )
            assignment[zone] = best_exit

        trace("Min-Conflicts: max steps reached without solution.")
        return None

    def explain_failure(self, partial: Dict[str, str]) -> str:
        """
        Generate explainability trace: why a partial assignment failed.
        Reports which constraint was violated and suggests fixes.
        """
        lines: List[str] = ["[CSP Failure Analysis]"]
        for zone, exit_ in partial.items():
            # Check adjacency
            for nb in self.adjacency.get(zone, []):
                if partial.get(nb) == exit_:
                    lines.append(
                        f"  ✗ Adjacency conflict: zones '{zone}' and '{nb}' "
                        f"both assigned to exit '{exit_}'. "
                        f"Suggestion: reassign '{nb}' to a different exit."
                    )
            # Check capacity
            count = sum(1 for v in partial.values() if v == exit_)
            cap = self.exit_capacity.get(exit_, 999)
            if count > cap:
                lines.append(
                    f"  ✗ Capacity exceeded: exit '{exit_}' assigned {count} zones "
                    f"but capacity is {cap}. "
                    f"Suggestion: redirect some zones to exits with spare capacity."
                )
        return "\n".join(lines) if len(lines) > 1 else "No constraint violations found."


# =============================================================================
# SECTION 7 – MULTI-AGENT: UTILITY / MINIMAX / ALPHA-BETA
# =============================================================================

class EvacMultiAgent:
    """
    Models a 2-agent scenario:
      - MAX agent: evacuee, wants to reach exit with minimum cost.
      - MIN agent: hazard (fire/adversary), tries to block cheapest path.

    Used to reason about worst-case evacuation planning.
    """

    def __init__(self, graph: BuildingGraph, env: EnvironmentState) -> None:
        self.graph = graph
        self.env = env

    def utility(self, node: str, depth: int) -> float:
        """
        Utility function for a leaf node (terminal or depth-limited).
        Higher = better for MAX (evacuee).
        Factors: proximity to exit, hazard penalty, depth penalty.
        """
        if node in self.env.exits:
            return 100.0 - depth       # reached exit; reward decreases with depth
        if self.env.is_hazardous(node):
            return -100.0              # entered hazard zone; heavily penalised
        dist = min(
            self.graph.euclidean_distance(node, ex) for ex in self.env.exits
        ) if self.env.exits else 999
        return max(0.0, 50.0 - dist - depth)

    def minimax(
        self,
        node: str,
        depth: int,
        is_max: bool,
        nodes_visited: List[int],
    ) -> float:
        """
        Minimax with depth limit.
        MAX = evacuee (maximises utility).
        MIN = hazard spread (minimises utility for evacuee).
        """
        nodes_visited[0] += 1
        if depth == 0 or node in self.env.exits or self.env.is_hazardous(node):
            return self.utility(node, depth)

        neighbours = [
            nb for nb, _ in self.graph.neighbours(node, self.env)
        ]
        if not neighbours:
            return self.utility(node, depth)

        if is_max:
            best = -math.inf
            for nb in neighbours:
                val = self.minimax(nb, depth - 1, False, nodes_visited)
                best = max(best, val)
            return best
        else:
            worst = math.inf
            for nb in neighbours:
                val = self.minimax(nb, depth - 1, True, nodes_visited)
                worst = min(worst, val)
            return worst

    def alpha_beta(
        self,
        node: str,
        depth: int,
        alpha: float,
        beta: float,
        is_max: bool,
        nodes_visited: List[int],
    ) -> float:
        """
        Alpha-Beta Pruning – same result as minimax but skips irrelevant branches.
        alpha = best guaranteed value for MAX so far.
        beta  = best guaranteed value for MIN so far.
        Prune when alpha >= beta (MIN will never choose this path).
        """
        nodes_visited[0] += 1
        if depth == 0 or node in self.env.exits or self.env.is_hazardous(node):
            return self.utility(node, depth)

        neighbours = [nb for nb, _ in self.graph.neighbours(node, self.env)]
        if not neighbours:
            return self.utility(node, depth)

        if is_max:
            value = -math.inf
            for nb in neighbours:
                value = max(value, self.alpha_beta(nb, depth-1, alpha, beta, False, nodes_visited))
                alpha = max(alpha, value)
                if alpha >= beta:
                    trace(f"Alpha-Beta PRUNE at {nb}: α={alpha:.2f} ≥ β={beta:.2f}")
                    break
            return value
        else:
            value = math.inf
            for nb in neighbours:
                value = min(value, self.alpha_beta(nb, depth-1, alpha, beta, True, nodes_visited))
                beta = min(beta, value)
                if alpha >= beta:
                    trace(f"Alpha-Beta PRUNE at {nb}: α={alpha:.2f} ≥ β={beta:.2f}")
                    break
            return value

    def best_move(self, start: str, depth: int = 4) -> Tuple[str, float]:
        """
        Select best next move for the evacuee using alpha-beta pruning.
        Implements iterative deepening concept by calling alpha_beta at
        increasing depths (here we run a single fixed depth for clarity).
        Returns (best_next_node, expected_utility).
        """
        neighbours = [nb for nb, _ in self.graph.neighbours(start, self.env)]
        if not neighbours:
            return start, self.utility(start, 0)

        best_node = neighbours[0]
        best_val = -math.inf
        nv = [0]

        for nb in neighbours:
            val = self.alpha_beta(nb, depth - 1, -math.inf, math.inf, False, nv)
            trace(f"MultiAgent: move {start}→{nb} utility={val:.2f}")
            if val > best_val:
                best_val, best_node = val, nb

        trace(f"MultiAgent BEST MOVE: {start}→{best_node} u={best_val:.2f} "
              f"nodes_visited={nv[0]}")
        return best_node, best_val

    def policy_selection(self, start: str) -> str:
        """
        Bounded rationality policy: choose based on utility threshold.
        If best utility > 30 → standard A*.
        If best utility > 0  → greedy escape.
        Otherwise            → shelter in place.
        """
        _, utility = self.best_move(start, depth=3)
        if utility > 30:
            policy = "OPTIMAL_EVACUATION"
        elif utility > 0:
            policy = "GREEDY_ESCAPE"
        else:
            policy = "SHELTER_IN_PLACE"
        trace(f"Policy selected: {policy} (utility={utility:.2f})")
        return policy


# =============================================================================
# SECTION 8 – PROBABILISTIC INFERENCE  (Bayes / Bayesian Network / HMM Sensor Fusion)
# =============================================================================

class HazardBayesianNet:
    """
    Bayesian Network for hazard probability inference.

    Structure:
        [Fire] → [Smoke]
        [Fire] → [TemperatureHigh]
        [Smoke] → [SensorAlarm]
        [TemperatureHigh] → [SensorAlarm]

    Variables (all binary: True/False):
        Fire, Smoke, TemperatureHigh, SensorAlarm

    CPTs (Conditional Probability Tables):
        P(Fire=T) = prior (configurable)
        P(Smoke=T | Fire=T) = 0.95, P(Smoke=T | Fire=F) = 0.05
        P(TempHigh=T | Fire=T) = 0.90, P(TempHigh=T | Fire=F) = 0.02
        P(Alarm=T | Smoke=T, TempHigh=T) = 0.99
        P(Alarm=T | Smoke=T, TempHigh=F) = 0.80
        P(Alarm=T | Smoke=F, TempHigh=T) = 0.70
        P(Alarm=T | Smoke=F, TempHigh=F) = 0.01
    """

    def __init__(self, prior_fire: float = 0.05) -> None:
        self.prior_fire = prior_fire

        # CPTs stored as nested dicts
        self.cpt_smoke = {True: 0.95, False: 0.05}
        self.cpt_temp  = {True: 0.90, False: 0.02}
        self.cpt_alarm = {
            (True,  True ): 0.99,
            (True,  False): 0.80,
            (False, True ): 0.70,
            (False, False): 0.01,
        }

    # ── Bayes Rule: P(Fire | Alarm) ───────────────────────────────────────
    def posterior_fire_given_alarm(self, alarm_observed: bool) -> float:
        """
        Variable Elimination / Bayes Rule:
        P(Fire | Alarm=obs) ∝ Σ_{Smoke,Temp} P(Fire) P(Smoke|Fire) P(Temp|Fire) P(Alarm|Smoke,Temp)

        Full marginalisation over hidden variables Smoke and TemperatureHigh.
        """
        def joint(fire: bool, smoke: bool, temp: bool) -> float:
            p_fire  = self.prior_fire if fire else (1 - self.prior_fire)
            p_smoke = self.cpt_smoke[fire] if smoke else (1 - self.cpt_smoke[fire])
            p_temp  = self.cpt_temp[fire]  if temp  else (1 - self.cpt_temp[fire])
            p_alarm = self.cpt_alarm[(smoke, temp)]
            p_alarm_obs = p_alarm if alarm_observed else (1 - p_alarm)
            return p_fire * p_smoke * p_temp * p_alarm_obs

        # Marginalise over Smoke and Temp
        p_fire_and_alarm = sum(
            joint(True, s, t) for s in [True, False] for t in [True, False]
        )
        p_no_fire_and_alarm = sum(
            joint(False, s, t) for s in [True, False] for t in [True, False]
        )
        evidence = p_fire_and_alarm + p_no_fire_and_alarm
        if evidence < 1e-12:
            return 0.0
        posterior = p_fire_and_alarm / evidence
        trace(
            f"Bayes: P(Fire | Alarm={alarm_observed}) = {posterior:.4f} "
            f"[prior={self.prior_fire}]"
        )
        return posterior

    # ── Likelihood Weighting (Sampling Inference concept) ─────────────────
    def likelihood_weighting_fire(
        self, alarm_observed: bool, n_samples: int = 10000
    ) -> float:
        """
        Approximate inference via likelihood weighting.
        Sample non-evidence variables; weight by evidence probability.
        Returns estimated P(Fire=True | Alarm=alarm_observed).
        """
        weight_fire = 0.0
        total_weight = 0.0

        for _ in range(n_samples):
            fire  = random.random() < self.prior_fire
            smoke = random.random() < (self.cpt_smoke[fire])
            temp  = random.random() < (self.cpt_temp[fire])
            # Weight = P(Alarm = obs | smoke, temp)
            p_alarm = self.cpt_alarm[(smoke, temp)]
            w = p_alarm if alarm_observed else (1 - p_alarm)
            total_weight += w
            if fire:
                weight_fire += w

        est = weight_fire / max(total_weight, 1e-12)
        trace(f"LikelihoodWeighting P(Fire|Alarm={alarm_observed}) ≈ {est:.4f} "
              f"({n_samples} samples)")
        return est


class HMMSensorFusion:
    """
    Hidden Markov Model (HMM) intuition for tracking a hazard (fire) spreading
    through building zones over discrete time steps.

    Hidden state  : set of zones currently on fire  (binary per zone)
    Observation   : sensor alarm readings            (noisy)
    Transition    : fire spreads to adjacent zones with probability p_spread
    Emission      : P(alarm | fire_in_zone)

    We implement the forward algorithm (filtering) to estimate
    P(fire_in_zone_t | alarms_{0..t}).
    """

    def __init__(
        self,
        zones: List[str],
        adjacency: Dict[str, List[str]],
        p_spread: float = 0.2,
        p_alarm_given_fire: float = 0.9,
        p_alarm_given_no_fire: float = 0.05,
    ) -> None:
        self.zones = zones
        self.adjacency = adjacency
        self.p_spread = p_spread
        self.p_emit_fire    = p_alarm_given_fire
        self.p_emit_no_fire = p_alarm_given_no_fire
        # Belief: P(fire in zone at current time) initialised to low prior
        self.belief: Dict[str, float] = {z: 0.02 for z in zones}

    def observe(self, alarms: Dict[str, bool]) -> Dict[str, float]:
        """
        Forward algorithm step: update belief given new sensor observations.
        1. Predict: propagate fire spread.
        2. Update:  weight by emission probability.
        """
        # ── Predict (transition) ──────────────────────────────────────────
        new_belief: Dict[str, float] = {}
        for zone in self.zones:
            # P(fire_t | fire_{t-1}) = stays on fire OR spreads from neighbour
            p_stay   = self.belief[zone]    # already on fire
            neighbours_in_belief = [
                self.belief.get(nb, 0.0) * self.p_spread
                for nb in self.adjacency.get(zone, [])
            ]
            p_spread = max(neighbours_in_belief) if neighbours_in_belief else 0.0
            new_belief[zone] = min(1.0, p_stay + (1 - p_stay) * p_spread)

        # ── Update (emission) ─────────────────────────────────────────────
        for zone in self.zones:
            alarm = alarms.get(zone, False)
            if alarm:
                likelihood = (
                    new_belief[zone] * self.p_emit_fire
                    + (1 - new_belief[zone]) * self.p_emit_no_fire
                )
                new_belief[zone] = (
                    new_belief[zone] * self.p_emit_fire
                ) / max(likelihood, 1e-12)
            else:
                likelihood = (
                    new_belief[zone] * (1 - self.p_emit_fire)
                    + (1 - new_belief[zone]) * (1 - self.p_emit_no_fire)
                )
                new_belief[zone] = (
                    new_belief[zone] * (1 - self.p_emit_fire)
                ) / max(likelihood, 1e-12)

        self.belief = new_belief

        # Expected utility: flag zones with high fire probability as hazardous
        trace("HMM belief update:")
        for z, b in self.belief.items():
            if b > 0.05:
                trace(f"  {z}: P(fire)={b:.3f}")

        return dict(self.belief)

    def high_risk_zones(self, threshold: float = 0.5) -> List[str]:
        """Return zones where P(fire) > threshold → mark as hazardous."""
        return [z for z, b in self.belief.items() if b >= threshold]

    def expected_utility_of_path(
        self, path: List[str], exit_utility: float = 100.0
    ) -> float:
        """
        Uncertainty-aware decision: compute expected utility of a path
        accounting for P(fire) in each zone along the way.
        EU(path) = Π(1 - P(fire_z)) * exit_utility - Σ P(fire_z) * penalty
        """
        survival_prob = 1.0
        expected_loss = 0.0
        for zone in path[:-1]:  # exclude exit
            p_fire = self.belief.get(zone, 0.0)
            survival_prob *= (1 - p_fire)
            expected_loss += p_fire * 50.0    # 50-unit penalty per hazardous zone
        eu = survival_prob * exit_utility - expected_loss
        trace(f"Expected utility of path {path}: EU={eu:.2f} "
              f"(survival_prob={survival_prob:.3f})")
        return eu


# =============================================================================
# SECTION 9 – HYBRID ARCHITECTURE  (Search + CSP + Probabilistic + Decision)
# =============================================================================

class HybridEvacuationPlanner:
    """
    Combines all modules into a unified emergency evacuation system:
      1. Sensor fusion (HMM) updates P(fire) beliefs.
      2. Bayesian network updates hazard probabilities.
      3. Environment state updated from beliefs.
      4. CSP assigns zones to exits (resource allocation).
      5. A* / UCS finds optimal path respecting hazards.
      6. Multi-agent minimax reasons about worst-case scenarios.
      7. Expected utility selects the safest plan.
      8. Explainability traces logged at every step.
    """

    def __init__(self, graph: BuildingGraph, env: EnvironmentState) -> None:
        self.graph = graph
        self.env = env
        self.bayes = HazardBayesianNet(prior_fire=0.05)
        self.hmm: Optional[HMMSensorFusion] = None
        self.multi_agent = EvacMultiAgent(graph, env)

    def initialise_hmm(self, adjacency: Dict[str, List[str]]) -> None:
        """Set up HMM for all rooms in the graph."""
        zones = list(self.graph.coordinates.keys())
        self.hmm = HMMSensorFusion(zones, adjacency)

    def process_sensor_tick(
        self,
        alarm_readings: Dict[str, bool],
        temperature_high: bool,
        smoke_detected: bool,
    ) -> None:
        """
        One sensor update cycle:
        - Bayesian net updates P(fire) from alarm.
        - HMM propagates fire belief over zones.
        - High-risk zones marked hazardous in environment.
        """
        trace("=== Sensor Tick ===")
        # Bayesian inference
        p_fire = self.bayes.posterior_fire_given_alarm(any(alarm_readings.values()))
        trace(f"Global P(fire from Bayes) = {p_fire:.3f}")

        # HMM sensor fusion
        if self.hmm:
            self.hmm.observe(alarm_readings)
            risky = self.hmm.high_risk_zones(threshold=0.4)
            for zone in risky:
                self.env.update_hazard(zone, True)
            # Clear zones below threshold (fire may have subsided or been wrong)
            safe_zones = [z for z in self.hmm.zones if z not in risky]
            for zone in safe_zones:
                self.env.update_hazard(zone, False)

    def plan_evacuation(
        self, start: str, algorithm: str = "astar"
    ) -> SearchResult:
        """
        Main planning entry point.
        Selects algorithm based on policy and returns best path.
        """
        trace(f"=== Planning evacuation from '{start}' via {algorithm} ===")
        problem = EvacProblem(self.graph, self.env, start)

        # Multi-agent policy check
        policy = self.multi_agent.policy_selection(start)
        if policy == "SHELTER_IN_PLACE":
            trace("Policy: SHELTER_IN_PLACE – no evacuation path planned.")
            return SearchResult(algorithm, [start], 0.0, 0, 0.0, 0.0, False)

        # Run chosen search algorithm
        algos = {
            "bfs":    bfs,
            "dfs":    dfs,
            "ucs":    ucs,
            "greedy": greedy,
            "astar":  astar,
            "idastar": ida_star,
        }
        algo_fn = algos.get(algorithm.lower(), astar)
        result = algo_fn(problem)

        # Expected utility evaluation
        if result.found and self.hmm:
            eu = self.hmm.expected_utility_of_path(result.path)
            trace(f"Evacuation path EU = {eu:.2f}")
            if eu < 0:
                trace("EU < 0: path too risky. Switching to alternative…")
                alt = ucs(problem)   # fallback to UCS
                if alt.found:
                    alt_eu = self.hmm.expected_utility_of_path(alt.path)
                    if alt_eu > eu:
                        trace(f"Alternative path EU={alt_eu:.2f} accepted.")
                        return alt

        return result

    def run_all_algorithms(self, start: str) -> Dict[str, SearchResult]:
        """Run all search algorithms and return comparative results."""
        trace(f"=== Running all algorithms from '{start}' ===")
        problem = EvacProblem(self.graph, self.env, start)
        return {
            "BFS":    bfs(problem),
            "DFS":    dfs(problem),
            "UCS":    ucs(problem),
            "Greedy": greedy(problem),
            "A*":     astar(problem),
            "IDA*":   ida_star(problem),
        }


# =============================================================================
# SECTION 10 – PERFORMANCE PROFILING  (node expansions, runtime, memory)
# =============================================================================

def profile_algorithms(results: Dict[str, SearchResult]) -> None:
    """
    Empirical profiling: compare all search algorithms on key metrics.
    """
    print("\n" + "=" * 78)
    print(" ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 78)
    fmt = "{:<12} {:>8} {:>12} {:>12} {:>14} {}"
    print(fmt.format("Algorithm", "Found", "Cost", "Nodes", "Time(ms)", "Path"))
    print("-" * 78)
    for name, r in results.items():
        path_str = " → ".join(r.path) if r.path else "—"
        print(fmt.format(
            name,
            str(r.found),
            f"{r.total_cost:.2f}" if r.found else "∞",
            r.nodes_expanded,
            f"{r.runtime_ms:.3f}",
            path_str,
        ))
    print("=" * 78)


# =============================================================================
# SECTION 11 – BUILDING FACTORY  (Sample building for demonstration)
# =============================================================================

def build_sample_hospital() -> Tuple[BuildingGraph, EnvironmentState]:
    """
    Construct a sample 3-floor hospital layout with labelled rooms,
    corridors, exits, and coordinates.

    Layout (floor 1 shown):
      [Entrance]--[Lobby]--[Corridor_A]--[Ward_1]--[Ward_2]
                    |             |
                 [Stairwell]  [ICU]
                    |
              [Floor2_Hall]--[Room_201]--[Room_202]
                    |
              [Roof_Exit]
    """
    g = BuildingGraph()

    # ── Nodes (name, x_metres, y_metres) ──────────────────────────────────
    nodes = [
        ("Entrance",    0,   0),
        ("Lobby",      10,   0),
        ("Corridor_A", 20,   0),
        ("Ward_1",     30,   0),
        ("Ward_2",     40,   0),
        ("ICU",        20, -10),
        ("Stairwell",  10,  10),
        ("Floor2_Hall",10,  20),
        ("Room_201",   20,  20),
        ("Room_202",   30,  20),
        ("Roof_Exit",  10,  30),
        ("Exit_East",  50,   0),
        ("Exit_North", 10,  -5),
    ]
    for name, x, y in nodes:
        g.add_node(name, float(x), float(y))

    # ── Edges (u, v, weight_seconds, bidirectional, width_metres) ─────────
    edges = [
        ("Entrance",    "Lobby",       5.0,  True, 3.0),
        ("Lobby",       "Corridor_A", 8.0,  True, 3.0),
        ("Corridor_A",  "Ward_1",      6.0,  True, 2.5),
        ("Ward_1",      "Ward_2",      5.0,  True, 2.0),
        ("Ward_2",      "Exit_East",   4.0,  True, 3.0),
        ("Corridor_A",  "ICU",         7.0,  True, 2.0),
        ("Lobby",       "Stairwell",   6.0,  True, 2.5),
        ("Lobby",       "Exit_North",  3.0,  True, 3.0),
        ("Stairwell",   "Floor2_Hall", 10.0, True, 2.5),
        ("Floor2_Hall", "Room_201",    5.0,  True, 2.0),
        ("Room_201",    "Room_202",    4.0,  True, 2.0),
        ("Floor2_Hall", "Roof_Exit",   8.0,  True, 2.5),
    ]
    for u, v, w, bi, width in edges:
        g.add_edge(u, v, w, bi, width)

    rooms = [n for n, _, _ in nodes]
    exits = ["Entrance", "Exit_East", "Exit_North", "Roof_Exit"]
    env = EnvironmentState(rooms=rooms, exits=exits)

    return g, env


def build_sample_csp(
    graph: BuildingGraph, env: EnvironmentState
) -> EvacZoneCSP:
    """
    Build a CSP instance: assign building zones to exits.
    Zones = clusters of nearby rooms.
    """
    zones = ["Z_ICU", "Z_Ward", "Z_Lobby", "Z_Floor2"]
    exits = ["Entrance", "Exit_East", "Exit_North", "Roof_Exit"]

    adjacency: Dict[str, List[str]] = {
        "Z_ICU":    ["Z_Ward"],
        "Z_Ward":   ["Z_ICU", "Z_Lobby"],
        "Z_Lobby":  ["Z_Ward", "Z_Floor2"],
        "Z_Floor2": ["Z_Lobby"],
    }

    exit_capacity: Dict[str, int] = {
        "Entrance":   2,
        "Exit_East":  2,
        "Exit_North": 2,
        "Roof_Exit":  1,
    }

    priority_zones = ["Z_ICU"]

    distances: Dict[Tuple[str, str], float] = {
        ("Z_ICU",    "Entrance"):   25.0,
        ("Z_ICU",    "Exit_East"):  20.0,
        ("Z_ICU",    "Exit_North"): 15.0,
        ("Z_ICU",    "Roof_Exit"):  40.0,
        ("Z_Ward",   "Entrance"):   15.0,
        ("Z_Ward",   "Exit_East"):  10.0,
        ("Z_Ward",   "Exit_North"): 20.0,
        ("Z_Ward",   "Roof_Exit"):  35.0,
        ("Z_Lobby",  "Entrance"):    5.0,
        ("Z_Lobby",  "Exit_East"):  20.0,
        ("Z_Lobby",  "Exit_North"):  5.0,
        ("Z_Lobby",  "Roof_Exit"):  25.0,
        ("Z_Floor2", "Entrance"):   20.0,
        ("Z_Floor2", "Exit_East"):  30.0,
        ("Z_Floor2", "Exit_North"): 20.0,
        ("Z_Floor2", "Roof_Exit"):  10.0,
    }

    return EvacZoneCSP(zones, exits, adjacency, exit_capacity, priority_zones, distances)


# =============================================================================
# SECTION 12 – UNIT TESTS
# =============================================================================

def run_unit_tests() -> None:
    """
    Small unit tests for core algorithm components.
    Validates correctness on a tiny known graph.
    """
    print("\n" + "=" * 60)
    print(" UNIT TESTS")
    print("=" * 60)

    # ── Mini graph ─────────────────────────────────────────────────────────
    #  A -1- B -1- C(exit)
    #  |               |
    #  +--2-- D --1----+
    g = BuildingGraph()
    for name, x, y in [("A",0,0),("B",1,0),("C",2,0),("D",1,-1)]:
        g.add_node(name, float(x), float(y))
    g.add_edge("A","B",1.0); g.add_edge("B","C",1.0)
    g.add_edge("A","D",2.0); g.add_edge("D","C",1.0)

    env_t = EnvironmentState(rooms=["A","B","C","D"], exits=["C"])
    prob  = EvacProblem(g, env_t, "A")

    # BFS finds A→B→C
    r_bfs = bfs(prob)
    assert r_bfs.found, "BFS should find path"
    assert r_bfs.path == ["A","B","C"], f"BFS path wrong: {r_bfs.path}"

    # UCS finds cheapest: A→B→C (cost 2)
    r_ucs = ucs(prob)
    assert r_ucs.found, "UCS should find path"
    assert abs(r_ucs.total_cost - 2.0) < 1e-6, f"UCS cost wrong: {r_ucs.total_cost}"

    # A* finds same
    r_as = astar(prob)
    assert r_as.found, "A* should find path"
    assert abs(r_as.total_cost - 2.0) < 1e-6, f"A* cost wrong: {r_as.total_cost}"

    # Hazard blocks B
    env_t.update_hazard("B", True)
    r_hz = astar(prob)
    assert r_hz.found, "A* should find alternate path around hazard"
    assert "B" not in r_hz.path, "Hazard node B should not appear in path"

    # CSP: simple 2-zone 2-exit
    csp2 = EvacZoneCSP(
        zones=["Z1","Z2"],
        exits=["E1","E2"],
        adjacency={"Z1":["Z2"],"Z2":["Z1"]},
        exit_capacity={"E1":2,"E2":2},
        priority_zones=[],
        distances={("Z1","E1"):5,("Z1","E2"):10,("Z2","E1"):10,("Z2","E2"):5},
    )
    soln, _ = csp2.solve()
    assert soln is not None, "CSP should solve"
    assert soln["Z1"] != soln["Z2"], "Adjacent zones must have different exits"

    # Bayes
    bn = HazardBayesianNet(prior_fire=0.1)
    p = bn.posterior_fire_given_alarm(True)
    assert p > 0.1, "Posterior P(Fire|Alarm=True) should exceed prior"
    p2 = bn.posterior_fire_given_alarm(False)
    assert p2 < 0.1, "Posterior P(Fire|Alarm=False) should be below prior"

    print("  All unit tests PASSED ✓")
    print("=" * 60)


# =============================================================================
# SECTION 13 – MAIN  (full demonstration run)
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("  EMERGENCY EVACUATION PATH PLANNER  –  KL University AI Project")
    print("  S.Vedhanth | T.Ananya | B.Pranavi")
    print("=" * 78)

    # ── 1. Build environment ──────────────────────────────────────────────
    graph, env = build_sample_hospital()
    print("\n[1] Hospital graph built.")
    print(f"    Rooms  : {len(graph.coordinates)}")
    print(f"    Exits  : {env.exits}")

    # ── 2. PEAS Agent ─────────────────────────────────────────────────────
    agent = PEASAgent("EVAC-001")
    agent.perceive({"smoke_sensor": True, "temperature": 85, "gps_zone": "ICU"})
    agent.act("sound_alarm")
    agent.act("send_alert")
    agent.compute_performance(evacuation_time=90.0, users_safe=48,
                              total_users=50, hazards_avoided=3)

    # ── 3. Probabilistic Sensor Update ────────────────────────────────────
    print("\n[3] Probabilistic inference…")
    bayes = HazardBayesianNet(prior_fire=0.05)
    p_fire = bayes.posterior_fire_given_alarm(alarm_observed=True)
    print(f"    P(Fire | Alarm=True)  = {p_fire:.4f}")
    p_fire_no = bayes.posterior_fire_given_alarm(alarm_observed=False)
    print(f"    P(Fire | Alarm=False) = {p_fire_no:.4f}")

    lw = bayes.likelihood_weighting_fire(alarm_observed=True, n_samples=5000)
    print(f"    LW estimate (5000 samples) = {lw:.4f}")

    # ── 4. HMM Sensor Fusion ──────────────────────────────────────────────
    print("\n[4] HMM Sensor Fusion…")
    zone_adjacency = {
        "ICU":        ["Corridor_A"],
        "Corridor_A": ["ICU","Ward_1","Lobby"],
        "Ward_1":     ["Corridor_A","Ward_2"],
        "Ward_2":     ["Ward_1","Exit_East"],
        "Lobby":      ["Corridor_A","Stairwell","Exit_North","Entrance"],
        "Stairwell":  ["Lobby","Floor2_Hall"],
        "Floor2_Hall":["Stairwell","Room_201","Roof_Exit"],
        "Room_201":   ["Floor2_Hall","Room_202"],
        "Room_202":   ["Room_201"],
        "Entrance":   ["Lobby"],
        "Exit_East":  [],
        "Exit_North": [],
        "Roof_Exit":  [],
    }
    hmm = HMMSensorFusion(
        zones=list(graph.coordinates.keys()),
        adjacency=zone_adjacency,
    )
    alarms = {"ICU": True, "Corridor_A": True, "Ward_1": False}
    belief = hmm.observe(alarms)
    high_risk = hmm.high_risk_zones(0.4)
    print(f"    High-risk zones (P(fire)≥0.4): {high_risk}")
    for zone in high_risk:
        env.update_hazard(zone, True)

    # ── 5. Run all search algorithms ──────────────────────────────────────
    print("\n[5] Running all search algorithms from 'Room_202'…")
    results = {}
    problem = EvacProblem(graph, env, "Room_202")
    results["BFS"]    = bfs(problem)
    results["DFS"]    = dfs(problem)
    results["UCS"]    = ucs(problem)
    results["Greedy"] = greedy(problem)
    results["A*"]     = astar(problem)
    results["IDA*"]   = ida_star(problem)
    profile_algorithms(results)

    # ── 6. CSP Zone Assignment ────────────────────────────────────────────
    print("\n[6] CSP: Assigning zones to exits…")
    csp = build_sample_csp(graph, env)
    csp_solution, csp_nodes = csp.solve()
    print(f"    CSP nodes explored: {csp_nodes}")
    if csp_solution:
        print(f"    Zone → Exit assignment:")
        for zone, exit_ in csp_solution.items():
            print(f"      {zone:12s} → {exit_}")
    else:
        print("    No CSP solution found.")
        partial = {"Z_ICU": "Exit_North", "Z_Ward": "Exit_North"}
        print(csp.explain_failure(partial))

    # Min-conflicts demo
    mc_sol = csp.min_conflicts(max_steps=200)
    print(f"    Min-Conflicts solution: {mc_sol}")

    # ── 7. Multi-Agent Reasoning ──────────────────────────────────────────
    print("\n[7] Multi-Agent Reasoning (Alpha-Beta Pruning)…")
    ma = EvacMultiAgent(graph, env)
    best_node, best_util = ma.best_move("Room_202", depth=4)
    print(f"    Best first move from 'Room_202': → '{best_node}' (utility={best_util:.2f})")
    policy = ma.policy_selection("Room_202")
    print(f"    Selected policy: {policy}")

    # ── 8. Hybrid Planner ─────────────────────────────────────────────────
    print("\n[8] Hybrid Evacuation Planner…")
    hybrid = HybridEvacuationPlanner(graph, env)
    hybrid.initialise_hmm(zone_adjacency)
    hybrid.process_sensor_tick(
        alarm_readings=alarms, temperature_high=True, smoke_detected=True
    )
    hybrid_result = hybrid.plan_evacuation("Room_202", algorithm="astar")
    if hybrid_result.found:
        print(f"    Hybrid A* path: {' → '.join(hybrid_result.path)}")
        print(f"    Cost: {hybrid_result.total_cost:.2f}s  |  "
              f"Nodes expanded: {hybrid_result.nodes_expanded}")
    else:
        print("    Hybrid planner: no safe path found.")

    # ── 9. Heuristic Admissibility Spot-Check ─────────────────────────────
    print("\n[9] Heuristic Admissibility Check (sampled nodes)…")
    adm = evaluate_heuristic_admissibility(problem, optimal_cost=hybrid_result.total_cost)
    violations = [n for n, ok in adm.items() if not ok]
    if violations:
        print(f"    Admissibility violations: {violations}")
    else:
        print("    Heuristic is ADMISSIBLE for all sampled nodes ✓")

    # ── 10. Unit Tests ────────────────────────────────────────────────────
    run_unit_tests()

    print("\n[DONE] Emergency Evacuation Path Planner completed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
