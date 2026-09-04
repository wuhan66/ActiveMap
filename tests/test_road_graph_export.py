from pathlib import Path

import numpy as np
from affine import Affine

from activemap.evaluation.road_graph import MunoRoadGraph, trace_skeleton


def test_trace_skeleton_preserves_branching_paths():
    skeleton = np.zeros((7, 7), dtype=np.uint8)
    skeleton[1:6, 3] = 1
    skeleton[3, 3:6] = 1

    paths = trace_skeleton(skeleton)

    covered_edges = sum(len(path) - 1 for path in paths)
    assert covered_edges >= 6
    assert any((3, 3) in path for path in paths)


def test_muno_graph_writer_emits_bidirectional_edges(tmp_path: Path):
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[2, 1:4] = 1
    graph = MunoRoadGraph()
    graph.add_skeleton(skeleton, Affine.translation(10, 20), simplify_tolerance=0)
    output = tmp_path / "1.graph"

    graph.save(output)

    vertex_text, edge_text = output.read_text(encoding="utf-8").split("\n\n")
    assert len(vertex_text.splitlines()) >= 2
    edges = {tuple(map(int, line.split())) for line in edge_text.splitlines() if line}
    assert all((right, left) in edges for left, right in edges)
