import itertools
from .utils import stage, done

def find_triangles(C):
    stage_time = stage("Finding triangles in crossings graph")
    triangles = []
    for i,j,k in itertools.combinations(C.nodes(),3):
        if C.has_edge(i,j) and C.has_edge(j,k) and C.has_edge(i,k):
            triangles.append((i,j,k))
    print(f"Found {len(triangles)} triangles in crossings graph")
    done(stage_time)
    return triangles