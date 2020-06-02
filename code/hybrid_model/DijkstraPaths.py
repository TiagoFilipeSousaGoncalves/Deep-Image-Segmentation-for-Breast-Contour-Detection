import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.interpolate as interpolate
import scipy.ndimage.morphology as morpho
from .priodict import priorityDictionary

alpha = 0.15
beta = 0.1
delta = 1.85
MAX_GRADIENT = 255
beta2 = 0.1

def transform(y):
    y -= y[0]
    angle = np.arctan2(y[-1,0], y[-1,1])
    M = cv2.getRotationMatrix2D((0,0),(angle*180)/np.pi,1)
    y = np.dot(y, M[0:2, 0:2])
    y /= np.sqrt(np.sum(y[-1]**2))
    return y

def create_mask(all_shapes):
    final_mask = np.zeros([1800, 2100])
    for y in all_shapes:
        y[:,0]*=(-1)
        y *= 1000
        y += [100, 500]
        y = np.array(spline(y)).transpose()
        for point in y:
            final_mask[tuple(point.astype(int))] = 1
    
    final_mask = morpho.binary_fill_holes(final_mask)
    b = morpho.binary_erosion(final_mask, structure=np.ones([3, 3]))
    x,y = np.nonzero(np.logical_and(final_mask,np.logical_not(b)))
    points = np.stack((x,y),axis=1).astype(float)
    points -= [100,500]
    points /= 1000
    return points
    
def dist_transform(points, end_points, flip, shape):
    external, internal = end_points
    angle = np.arctan2(external[0] - internal[0], external[1] - internal[1])    
    scale = np.sqrt(np.sum(external-internal**2))
    dislocation = external
    if flip:
        points[:,1] *= -1
    points *= scale
    M = cv2.getRotationMatrix2D((0, 0), (angle*180)/np.pi, 1)
    points = np.dot(points, M)
    points += dislocation
    mask = np.zeros(shape)
    mask[points[:, 0], points[:, 1]] = 1
    mask = morpho.binary_fill_holes(mask)
    mask = morpho.distance_transform_edt(mask)
    return points


def spline(points, n_points=10000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out


def adjust(points, p1, p2, flip=False):
    if flip:
        points[:,1]*=-1
        offset = p2
    else:
        offset = p1
    scale = np.sqrt(np.sum((p1-p2)**2))
    angle = -np.arctan2(*(p2-p1))
    M = cv2.getRotationMatrix2D((0, 0), (angle*180)/np.pi, 1)[0:2,0:2]
    points *= scale
    points = np.dot(points, M)
    points += offset
    return points

"""
_______________________________________________________________________________
        SHORTEST PATH FOR ARBITRARY GRAPHS
_______________________________________________________________________________
"""

def Dijkstra(G,start,end=None):
    """
    Find shortest paths from the start vertex to all
    vertices nearer than or equal to the end.

    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.

    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.

    The output is a pair (D,P) where D[v] is the distance
    from start to v and P[v] is the predecessor of v along
    the shortest path from s to v.

    Dijkstra's algorithm is only guaranteed to work correctly
    when all edge lengths are positive. This code does not
    verify this property for all edges (only the edges seen
    before the end vertex is reached), but will correctly
    compute shortest paths even for some graphs with negative
    edges, and will raise an exception if it discovers that
    a negative edge has caused it to make a mistake.
    """

    D = {}	 # dictionary of final distances
    P = {}	 # dictionary of predecessors
    Q = priorityDictionary()   # est.dist. of non-final vert.
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        if v == end: break

        for w in G[v]:
            vwLength = D[v] + G[v][w]
            if w in D:
                if vwLength < D[w]:
                    print("Dijkstra: found better path to already-final vertex")
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v

    return (D, P)


def shortestPath(G, start, end):
    """
    Find a single shortest path from the given start vertex
    to the given end vertex.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """

    D, P = Dijkstra(G, start, end)

    
    Path = []
    while 1:
        Path.append(end)
        if end == start: break
        end = P[end]
    Path.reverse()
    return Path


def build_graph(magnitude, subimage):
        
    G = dict()
    for i in range(subimage[0],subimage[2]):
        for j in range(subimage[1],subimage[3]):
            G[(i,j)] = dict()
            other_vertex = [(i+1, j-1), (i+1, j), (i+1, j+1),
                            (i, j-1),             (i, j+1),
                            (i-1, j-1), (i-1, j), (i-1, j+1)
                           ]
            # If in bounds
            for v in other_vertex:
                if v[0] >= subimage[0] and v[0] < subimage[2] and\
                   v[1] >= subimage[1] and v[1] < subimage[3]:
                       G[(i, j)][(v[0], v[1])] = dist(magnitude, (i, j), v)
    return G

def dist(weight, u, v):

    M, transformed_model = weight
    prior = max(transformed_model[u], transformed_model[v])

    d = np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2)
    z = MAX_GRADIENT-min(M[u], M[v])

    f_ = d*(alpha*np.exp(beta*z+beta2*prior)+delta)
    return f_


#all_shapes = list()
#for curr_y in Y:
#    for side in ["left", "right"]:
#        if side is "left":
#            y = curr_y[0:34]
#            y = np.reshape(y, [-1, 2])
#        else:
#            y = curr_y[34:68]
#            y = np.reshape(y, [-1, 2])
#            y[:, 0] *= (-1)

#        y = transform(y)
#        all_shapes.append(y)