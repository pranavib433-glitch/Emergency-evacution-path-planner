from flask import Flask, jsonify, request, send_from_directory
from evacuation_planner import (
    build_sample_hospital,
    EvacProblem,
    astar, bfs, ucs,
    HybridEvacuationPlanner,
    EvacZoneCSP,
    build_sample_csp,
)

app = Flask(__name__, static_folder="static")

# Build the graph once when server starts
graph, env = build_sample_hospital()
planner = HybridEvacuationPlanner(graph, env)

# ── Serve the frontend ────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ── API: get shortest path ────────────────────────────────────
@app.route("/api/path", methods=["GET"])
def get_path():
    start     = request.args.get("start", "Room_202")
    algorithm = request.args.get("algo",  "astar")
    result    = planner.plan_evacuation(start, algorithm)
    return jsonify({
        "path":       result.path,
        "cost":       result.total_cost,
        "nodes":      result.nodes_expanded,
        "found":      result.found,
        "algorithm":  result.algorithm,
    })

# ── API: update hazard ────────────────────────────────────────
@app.route("/api/hazard", methods=["POST"])
def update_hazard():
    data = request.json
    node      = data.get("node")
    hazardous = data.get("hazardous", True)
    env.update_hazard(node, hazardous)
    return jsonify({"status": "updated", "node": node, "hazardous": hazardous})

# ── API: get all rooms ────────────────────────────────────────
@app.route("/api/rooms", methods=["GET"])
def get_rooms():
    return jsonify({
        "rooms": list(graph.coordinates.keys()),
        "exits": env.exits,
    })

if __name__ == "__main__":
    app.run(debug=True)