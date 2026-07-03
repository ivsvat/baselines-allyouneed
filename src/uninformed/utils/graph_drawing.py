import cv2
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_and_draw_graph(
    node_positions,
    k,
    coord_x0y0,
    source_size: tuple[int, int],
    line_thickness: int,
    output_size=(224, 224),
) -> np.ndarray:
    n = len(node_positions)
    if n == 0:
        np.zeros((output_size[1], output_size[0], 1), np.uint8)
    n_neighbors = min(k + 1, n)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(node_positions)
    _, indices = knn.kneighbors(node_positions)
    indices = indices[:, 1:].astype(int)

    G = nx.Graph()
    pos = {i: (float(x), float(y)) for i, (x, y) in enumerate(node_positions)}
    G.add_nodes_from(pos.keys())
    nx.set_node_attributes(G, pos, "pos")
    edges = {
        (min(i, j), max(i, j))
        for i, neigh in enumerate(indices)
        for j in neigh
        if i != j
    }
    G.add_edges_from(edges)

    H, W = output_size

    pos = nx.get_node_attributes(G, "pos")
    if not pos or G.number_of_edges() == 0:
        return np.zeros((H, W), np.uint8)

    x0, y0 = float(coord_x0y0[0]), float(coord_x0y0[1])  # coords are (x, y)

    thickness = line_thickness

    nodes = list(pos.keys())
    pts = np.array(object=[pos[n] for n in nodes], dtype=np.float32)  # (x, y)
    px = np.rint(pts[:, 0] - x0).astype(int)
    py = np.rint(pts[:, 1] - y0).astype(int)
    id2pt = {n: (int(px[i]), int(py[i])) for i, n in enumerate(nodes)}

    img = np.zeros((source_size[1], source_size[0], 1), np.uint8)
    for u, v in G.edges():
        p1 = id2pt.get(u)
        p2 = id2pt.get(v)
        if p1 is not None and p2 is not None:
            cv2.line(
                img,
                p1,
                p2,
                color=255,  # type: ignore
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    interp = cv2.INTER_LINEAR
    img = cv2.resize(img, (W, H), interpolation=interp)
    return img


def draw_graph_to_u8(
    graph: nx.Graph,
    coord_x0y0: np.ndarray,
    line_thickness: int,
    source_size: tuple[int, int],
    output_size=(224, 224),
) -> np.ndarray:
    H, W = output_size

    pos = nx.get_node_attributes(graph, "pos")
    if not pos or graph.number_of_edges() == 0:
        return np.zeros((H, W), np.uint8)

    x0, y0 = float(coord_x0y0[0]), float(coord_x0y0[1])  # coords are (x, y)

    thickness = line_thickness

    nodes = list(pos.keys())
    pts = np.array(object=[pos[n] for n in nodes], dtype=np.float32)  # (x, y)
    px = np.rint(pts[:, 0] - x0).astype(int)
    py = np.rint(pts[:, 1] - y0).astype(int)
    id2pt = {n: (int(px[i]), int(py[i])) for i, n in enumerate(nodes)}

    img = np.zeros((source_size[1], source_size[0], 1), np.uint8)
    for u, v in graph.edges():
        p1 = id2pt.get(u)
        p2 = id2pt.get(v)
        if p1 is not None and p2 is not None:
            cv2.line(
                img,
                p1,
                p2,
                color=255,  # type: ignore
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    interp = cv2.INTER_LINEAR
    img = cv2.resize(img, (W, H), interpolation=interp)
    return img


def prepare_graph_from_nodes(node_positions: np.ndarray, k: int = 5) -> nx.Graph:
    n = len(node_positions)
    if n == 0:
        return nx.Graph()
    n_neighbors = min(k + 1, n)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(node_positions)
    _, indices = knn.kneighbors(node_positions)
    indices = indices[:, 1:].astype(int)

    G = nx.Graph()
    pos = {i: (float(x), float(y)) for i, (x, y) in enumerate(node_positions)}
    G.add_nodes_from(pos.keys())
    nx.set_node_attributes(G, pos, "pos")
    edges = {
        (min(i, j), max(i, j))
        for i, neigh in enumerate(indices)
        for j in neigh
        if i != j
    }
    G.add_edges_from(edges)
    return G
