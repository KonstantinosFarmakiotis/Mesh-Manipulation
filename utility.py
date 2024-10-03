import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import cm
from scipy.sparse import csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig,  lstsq
from scipy.optimize import minimize
import scipy.spatial.transform as st

def k_ring_adjacency(idx, triangles, k=1, num_vertices=None):

    adj_matrix = adjacency_matrix(triangles, num_vertices)

    #kth power
    adj_matrix = adj_matrix ** k

    #neighbors of specified index are the indices of non-zero elements of that row
    neighbors = adj_matrix[idx, :].toarray()

    return neighbors.nonzero()[1]

def adjacency_matrix(triangles, num_vertices = None):

    if num_vertices is None:
        num_vertices = triangles.max()+1

    #initializing sparse matrix
    adj_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.uint16)

    #iterating triangles to populate the array
    for tri in triangles:

        v1, v2, v3 = tri
        adj_matrix[v1, v2] = 1
        adj_matrix[v2, v1] = 1
        adj_matrix[v2, v3] = 1
        adj_matrix[v3, v2] = 1
        adj_matrix[v3, v1] = 1
        adj_matrix[v1, v3] = 1

    #converting to csr
    adj_matrix = adj_matrix.tocsr()

    return adj_matrix

def degree_matrix(adj, exponent=1):

    num_vertices = adj.shape[0]
    diagonals = np.zeros(num_vertices)

    if exponent==1:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum()
        return diags(diagonals, format="csr", dtype=np.int32)
    else:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum().astype(np.float32)**exponent
        return diags(diagonals, format="csr", dtype=np.float32)

def delta_coordinates(vertices, triangles, use_laplacian=True):

    if use_laplacian:
        L = random_walk_laplacian(triangles)
        delta = L @ vertices
    else:
        delta = np.zeros_like(vertices)
        for i, vi in enumerate(vertices):
            neighbors = k_ring_adjacency(i, triangles, 1)
            delta[i] = vi - vertices[neighbors, :].mean(0)
    return delta

def graph_laplacian(triangles):

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    D = degree_matrix(A, exponent=1)

    L = D - A

    return L

def random_walk_laplacian(triangles, subtract=True):

    num_vertices = triangles.max()+1

    A = adjacency_matrix(triangles, num_vertices=num_vertices)
    Dinv = degree_matrix(A, exponent=-1)
    
    if subtract:
        L = eye(num_vertices, num_vertices, 0) - Dinv @ A
    else:
        L = Dinv @ A
        
    return L

def sample_colormap(scalars, name="inferno"):

    avail_maps = ["inferno", "magma", "viridis", "cividis"]

    if name not in avail_maps:
        warnings.warn(f"Only {avail_maps} colormaps are supported. Using inferno.")
        name = "inferno"

    colormap = cm.get_cmap(name, 12)
    colors = colormap(scalars)

    return colors[:,:-1]



        #  ================          ERGASIA             ==================
        
        
# erotima a
def laplacian_smoothing(triangles, vertices, l=0.5):

    delta = delta_coordinates(vertices, triangles, use_laplacian=True)
    vertices -= l*delta
    return vertices

def taubin_smoothing(triangles, vertices, l=0.5, m=0.5):

    delta = delta_coordinates(vertices, triangles, use_laplacian=True)
    vertices -= l*delta
    delta = delta_coordinates(vertices, triangles, use_laplacian=True)
    vertices += m*delta
    return vertices


# erotima b
def taubin_smoothing_local(triangles, vertices, neighbors, l=0.5, m=0.5):

    vertices_new = np.array(vertices)
    delta = delta_coordinates(vertices_new, triangles, use_laplacian=True)
    vertices_new[neighbors] -= l*delta[neighbors]
    delta = delta_coordinates(vertices_new, triangles, use_laplacian=True)
    vertices_new[neighbors] += m*delta[neighbors]
    return vertices_new


# erotima c
def skeletonization_adjacency_matrix(vertices, triangles, num_vertices = None):

    if num_vertices is None:
        num_vertices = triangles.max()+1

    #initializing sparse matrix
    adj_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.uint16)

    #iterating triangles to populate the array
    for tri in triangles:
        v1, v2, v3 = tri
        vert_1 = vertices[v1]
        vert_2 = vertices[v2]
        vert_3 = vertices[v3]

        side_12 = (np.linalg.norm(vert_1-vert_2)) 
        side_23 = (np.linalg.norm(vert_2-vert_3)) 
        side_13 = (np.linalg.norm(vert_1-vert_3))
         
        angle_1 = np.arccos((side_12**2 + side_13**2 - side_23**2) / (2 * side_12 * side_13))
        angle_2 = np.arccos((side_12**2 + side_23**2 - side_13**2) / (2 * side_12 * side_23))
        angle_3 = np.arccos((side_23**2 + side_13**2 - side_12**2) / (2 * side_13 * side_23))
        
        adj_matrix[v1, v2] += 1 / np.tan(np.radians(angle_3)) 
        adj_matrix[v2, v1] += 1 / np.tan(np.radians(angle_3)) 
        adj_matrix[v2, v3] += 1 / np.tan(np.radians(angle_1)) 
        adj_matrix[v3, v2] += 1 / np.tan(np.radians(angle_1)) 
        adj_matrix[v3, v1] += 1 / np.tan(np.radians(angle_2)) 
        adj_matrix[v1, v3] += 1 / np.tan(np.radians(angle_2)) 

    #converting to csr
    adj_matrix = adj_matrix.tocsr()
    return adj_matrix

def skeletonization_laplacian(vertices, triangles):

    num_vertices = triangles.max()+1

    A = skeletonization_adjacency_matrix(vertices, triangles, num_vertices=num_vertices)
    D = degree_matrix(A, exponent=1)

    L = (D - A)
    
    return L

def area_of_triangle(vertices, triangles):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    area = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    return area

# def f(x, WL, L, WH, vertices):
    
#     x = x.reshape(vertices.shape)
#     return np.power(np.linalg.norm(np.dot(np.dot(WL,L),x)),2
#                     ) + np.sum(np.dot(np.power(WH,2) , np.power(np.linalg.norm(x-vertices, axis=1),2)))

def skeletonization(triangles, vertices, WL, WH, iterator, vertex_areas_original, neighbors = np.array([])):
    
    if iterator == 0 :
        
        WL = np.diag(np.full(vertices.shape[0], 0.1))
        WH = np.diag(np.full(vertices.shape[0], 1))
        
        vertex_areas_original = np.zeros((vertices.shape[0],1))
        for vertex_index in range(len(vertices)):
            vertex_triangles = triangles[np.any(triangles == vertex_index, axis=1)]
            vertex_areas_original[vertex_index] = np.sum(area_of_triangle(vertices, vertex_triangles))
        
        print("turn: ", iterator+1)
        iterator = 1
    
    else :
        
        sl = 2
        WL = sl*WL
        
        vertex_areas = np.zeros((vertices.shape[0],1))
        for vertex_index in range(len(vertices)):
            vertex_triangles = triangles[np.any(triangles == vertex_index, axis=1)]
            vertex_areas[vertex_index] = np.sum(area_of_triangle(vertices, vertex_triangles))
        
        sH = np.sqrt(np.absolute(np.divide(vertex_areas_original,vertex_areas)))
        WH = np.multiply(sH,WH)
        
        print("turn: ", iterator+1)
        iterator += 1
    
    L = skeletonization_laplacian(vertices, triangles).toarray()
    
    # print("vertices: ", vertices.shape, "triangles: ", triangles.shape)
    # print("WL: ", WL.shape, "L: ", L.shape, "WH: ", WH.shape)
    
    WLL = np.dot(WL,L)
    WHv = np.dot(WH,vertices)
    A = np.concatenate((WLL,WH), axis=0)
    b = np.concatenate((np.zeros_like(WHv), WHv), axis=0)
    
    A = np.where(np.isnan(A) | np.isinf(A), 0, A)
    b = np.where(np.isnan(b) | np.isinf(b), 0, b)
    # print("A: ", A.shape, "b: ", b.shape)
    
    vertices_new = lstsq(A, b)[0]
    
    if neighbors != np.array([]) :
        vertices[neighbors] = vertices_new[neighbors]
        vertices_new = vertices 
    
    # vertices_new = np.empty(vertices.shape)
    # vertices_new = minimize(f, vertices_new, args=(WL,L,WH,vertices)).x
    # vertices_new = vertices_new.reshape(vertices.shape)
    
    return vertices_new, WL, WH, iterator, vertex_areas_original


#erotima e
def shortest_edge(triangles, vertices, num):
    
    v1, v2, v3 = vertices[triangles][:,0],vertices[triangles][:,1],vertices[triangles][:,2]
    edge_lengths1 = np.linalg.norm((v1 - v2), axis=1)
    edge_lengths2 = np.linalg.norm((v2 - v3), axis=1)
    edge_lengths3 = np.linalg.norm((v1 - v3), axis=1)
    
    edge_data1 = np.concatenate((edge_lengths1.reshape(-1,1), v1,v2), axis=1)
    edge_data2 = np.concatenate((edge_lengths2.reshape(-1,1), v2,v3), axis=1)
    edge_data3 = np.concatenate((edge_lengths3.reshape(-1,1), v1,v3), axis=1)
    
    edge_data = np.concatenate((edge_data1, edge_data2, edge_data3), axis=0) 
    sorted_indices = np.argsort(edge_data[:,0])

    # partition the array based on the sorted indices
    partitioned_data = edge_data[sorted_indices,:][:num]
    
    
    return partitioned_data


def edge_collapse(triangles, vertices, num):
    
    edges = shortest_edge(triangles, vertices, num)
    v1 = edges[:,1:4]
    v2 = edges[:,4:7]
    new_vertices = (v1 + v2)/2 
    
    for tri in triangles:
        for i in range(len(v1)):
            if np.isin(v1[i], vertices[tri]).all():
                if np.isin(v2[i], vertices[tri]).all():
                    index = np.where(np.all(tri == triangles, axis=1))[0]
                    triangles = np.delete(triangles, index, axis=0)

    for i,new_vertex in enumerate(new_vertices):
        vertices = np.where(np.logical_or(vertices == v1[i], vertices == v2[i]), new_vertex, vertices)
        
    return vertices,triangles

def edge_create(triangles, vertices):
    
    v1, v2, v3 = vertices[triangles][:,0],vertices[triangles][:,1],vertices[triangles][:,2]
    centroids = np.empty((0,3))
    
    for i in range(len(triangles)):
        centroids = np.append(centroids, np.mean(vertices[triangles][i], axis=0))
    
    centroids = centroids.reshape((len(triangles),3))
    
    vertices_new = np.concatenate((vertices, centroids), axis=0)
    
    
    triangles_new = np.empty((0,3))
    for i,tri in enumerate(triangles):
        triangles_new = np.append(triangles_new, [tri[0],tri[1],len(vertices_new)-len(triangles)+i]) 
        triangles_new = np.append(triangles_new, [tri[1],tri[2],len(vertices_new)-len(triangles)+i])
        triangles_new = np.append(triangles_new, [tri[0],tri[2],len(vertices_new)-len(triangles)+i])

    
    triangles_new = triangles_new.reshape((int(len(triangles_new)/3), 3)).astype(int)
    
    return vertices_new, triangles_new

#erotima f
def extract_coating(triangles, vertices, neighbors, selected_vertex):
    
    # Sort neighbors on y axis
    y_values = vertices[neighbors, 1]
    sorted_y_values = np.argsort(y_values, axis=0)
    neighbors = np.reshape(neighbors[sorted_y_values], (len(neighbors)))
    
    vertices_new = taubin_smoothing_local(triangles, vertices, neighbors, 0.7, 0.3)
    for i in range(10):
        vertices_new = taubin_smoothing_local(triangles, vertices_new, neighbors, 0.7, 0.3)
    
    coat = np.subtract(vertices[neighbors], vertices_new[neighbors])  
    coat = np.reshape(coat, (len(coat),3))
    
    first_vector = np.subtract(vertices[selected_vertex], vertices_new[selected_vertex])
    return coat, first_vector

def coat_rotation(selected_vector1, selected_vector2, coat):
    
    # Coat rotation to match the new surface orientation
    v1 = selected_vector1
    v2 = selected_vector2
    
    # Find the rotation vector that aligns a to b
    v = np.cross (v1, v2)
    s = np.linalg.norm (v)
    c = np.dot (v1, v2)
    angle = np.arctan2 (s, c)
    rotvec = angle * v / s

    # Create a rotation object from the rotation vector
    r = st.Rotation.from_rotvec (rotvec)

    # Apply the rotation to coat
    coat = r.apply(coat)
    
    return coat

def apply_coating(triangles, vertices, neighbors, coat, first_vector, selected_vertex):
    
    # Sort neighbors on y axis
    y_values = vertices[neighbors, 1]
    sorted_y_values = np.argsort(y_values, axis=0)
    neighbors = np.reshape(neighbors[sorted_y_values], (len(neighbors)))
    
    n = np.minimum(len(coat), len(neighbors))
    vertices_new = taubin_smoothing_local(triangles, vertices, neighbors, 0.5, 0.5)
    for i in range(4):
        vertices_new = taubin_smoothing_local(triangles, vertices_new, neighbors, 0.5, 0.5)
        
    # print("\nmodel old\n", vertices[neighbors[0:15]], "\nmodel new\n", vertices_new[neighbors[0:15]], "\n")

    second_vector = np.subtract(vertices[selected_vertex], vertices_new[selected_vertex])
    
    # Rotate coat to match the orientation of the second vector
    coat = coat_rotation(first_vector, second_vector, coat)
    
    norm_coat = np.linalg.norm(coat[0])
    norm_new = np.linalg.norm(vertices_new[neighbors[0]]-vertices[neighbors[0]])
    # coat = np.multiply(5*norm_new/norm_coat, coat)
    # coat = np.multiply(2, coat)

    vertices_new[neighbors[:n]] = np.add(vertices_new[neighbors[:n]], coat[:n])
    
    # print("\nmodel + coat new\n", vertices_new[neighbors[0:15]], "\n")
    

    # print(norm_new/norm_coat)
    # print("\ncoat new\n", coat[0:15])
    # print(neighbors[:n])
    # print("\ndelta", delta[neighbors[:n]])
    # print("\ncoat", coat[:n])
    return vertices_new

    
    #  ================          TELOS         ERGASIAS             ==================