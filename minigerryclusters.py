import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
try:
    import Atlas
except ImportError:
    print("The 'Atlas' library is required for some functions")
    raise ImportError("Atlas not found")
from scipy.optimize import linear_sum_assignment as LSA
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.colors import ListedColormap
import os
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

#color map for Political Geometry book
colors = [[0.8,0.392156862745098,0.325490196078431],
[0.862745098039216,0.713725490196078,0.274509803921569],
[0.36078431372549,0.682352941176471,0.611764705882353],
[0.72156862745098,0.580392156862745,0.713725490196078],
[0.470588235294118,0.419607843137255,0.607843137254902],
[0.313725490196078,0.0588235294117647,0.23921568627451],
[0.803921568627451,0.850980392156863,0.376470588235294],
[0.329411764705882,0.47843137254902,0.250980392156863],
[0.352941176470588,0.694117647058824,0.803921568627451],
[0.682352941176471,0.549019607843137,0.380392156862745],
[0.701960784313725,0.733333333333333,0.823529411764706],
[0.372549019607843,0.376470588235294,0.411764705882353],
[0.701960784313725,0.933333333333333,0.756862745098039],
[0.588235294117647,0.666666666666667,0.290196078431373],
[0.556862745098039,0.16078431372549,0.0980392156862745],
[0.250980392156863,0.376470588235294,0.803921568627451],
[0.92156862745098,0.909803921568627,0.776470588235294],
[0.866666666666667,0.6,0.368627450980392],
[0.368627450980392,0.552941176470588,0.552941176470588],
[0.207843137254902,0.231372549019608,0.498039215686275]]
gerrybook_cmap = matplotlib.colors.ListedColormap(colors)
mycmap = matplotlib.colors.ListedColormap(colors)

def read_atlas_and_count_maps(file_path):
    """Read an Atlas file and count the number of redistricting plans it contains.
    
    Args:
        file_path (str): Path to the Atlas file
    
    Returns:
        tuple: (atlas, count) where atlas is the opened Atlas object and count is the number of plans
    """
    atlas = Atlas.openAtlas(file_path)
    count = 0
    mp = Atlas.nextMap(atlas)
    while mp is not None:
        count += 1
        mp = Atlas.nextMap(atlas)
    Atlas.closeAtlas(atlas)
    atlas = Atlas.openAtlas(file_path)
    return atlas, count

def atlas_to_binary_vectors(filepath, nodes, k):
    """Convert Atlas redistricting plans to binary district membership vectors.
    
    Args:
        filepath (str): Path to the Atlas file
        nodes (list): List of node dictionaries with 'county' and 'prec_id' keys
        k (int): Number of districts in each plan
    
    Returns:
        np.ndarray: Boolean array of shape (num_plans, k, num_nodes) representing district membership
    """
    atlas, count = read_atlas_and_count_maps(filepath)
    districts = []
    mp = Atlas.nextMap(atlas)
    for i in tqdm(range(count)):
        districts.append(get_binary_vectors(mp, nodes, k))
        mp = Atlas.nextMap(atlas)
    return np.array(districts, dtype=bool)

def get_binary_vectors(map, nodes, k):
    """Convert a single redistricting plan to binary district membership vectors.
    
    Args:
        map: Atlas map object containing district assignments
        nodes (list): List of node dictionaries with 'county' and 'prec_id' keys
        k (int): Number of districts
    
    Returns:
        np.ndarray: Boolean array of shape (k, num_nodes) where each row represents a district
    """
    vs = [np.zeros(len(nodes)) for i in range(k)]
    for n in range(len(nodes)):
        #whole county
        county_string = "[\"{}\"]".format(nodes[n]['county'])
        if county_string in map.districting:
            d = map.districting[county_string]
            vs[d-1][n] = 1
        else:
            county_string = "[\"{}_{}\"]".format(
                nodes[n]['county'],
                nodes[n]['prec_id']
            )
            d = map.districting[county_string]
            vs[d-1][n] = 1
    return np.array(vs, dtype=bool)

def match_families_to_means(families, means):
    """Assign each redistricting plan to its nearest cluster centroid.
    
    Args:
        families (list): List of redistricting plans
        means (list): List of cluster centroids
    
    Returns:
        list: Cluster assignment (index) for each plan
    """
    unrolled_families = np.array([d for p in families for d in p])
    unrolled_means = np.array([d for p in means for d in p])
    all_sq_distances = euclidean_distances(unrolled_means, unrolled_families)**2
    k = len(families[0])

    def get_indexing(plan_idx, mean_idx):
        M = all_sq_distances[mean_idx*k:(mean_idx+1)*k,:][:,plan_idx*k:(plan_idx+1)*k]
        row_ind, col_ind = LSA(M)
        return col_ind, sum(M[r,c] for r, c in zip(row_ind, col_ind))**(1/2)
    
    K = len(means)
    distances = [
        [get_indexing(i, j)[1] for j, mean in enumerate(means) ] for i, p in enumerate(families)
    ]
    matching = [np.argmin(distances[i]) for i, p in enumerate(families)]
    return matching

def distance_function(u, v, ord=2):
    """Calculate the distance between two vectors.
    
    Args:
        u (np.ndarray): First vector
        v (np.ndarray): Second vector
        ord (int): Order of the norm (default: 2 for Euclidean distance)
    
    Returns:
        float: Distance between the vectors
    """
    return np.linalg.norm(u-v, ord=ord)

def mycmap_alpha(alpha, K, colors=None):
    """Create an RGBA color tuple with specified transparency for visualization.
    
    Args:
        alpha (float): Alpha transparency value between 0 (transparent) and 1 (opaque)
        K (int): Color index to select from the colormap
        colors (np.ndarray, optional): Array of color indices. If None, uses sequential integers
    
    Returns:
        tuple: RGBA color tuple (R, G, B, alpha) with values in [0, 1]
    """
    if colors is None:
        colors = np.arange(K+1)
    R = list(mycmap(colors[K]))
    R[-1] = alpha
    return tuple(R)

def get_mean_vectors(vectors, num_precincts):
    """Calculate the mean of a collection of district vectors.
    
    Args:
        vectors (list): List of district vectors to average
        num_precincts (int): Number of precincts (length of each vector)
    
    Returns:
        np.ndarray: Mean vector, or zero vector if input list is empty
    """
    if len(vectors) == 0:
        return np.zeros(num_precincts)
    return np.mean(vectors, axis=0)

def k_means(families, num_precincts=None, q=2, K=1, eps=1e-7, max_iter=1000, initial_means=None, verbose=False, return_distances=False, seed=2026):
    """Perform k-means clustering on redistricting plans with district alignment.
    
    This function clusters redistricting plans while accounting for district label
    permutations using the Hungarian algorithm (linear sum assignment). Each plan
    consists of k district vectors, and the algorithm finds optimal alignments
    between plans and cluster centroids.
    
    Args:
        families (list): List of redistricting plans, where each plan is a list of k district vectors
        num_precincts (int, optional): Number of precincts (length of district vectors). If None, inferred from data
        q (int): Norm parameter for distance calculation (default: 2 for Euclidean)
        K (int): Number of clusters (default: 1)
        eps (float): Convergence threshold for centroid movement (default: 1e-7)
        max_iter (int): Maximum number of iterations (default: 1000)
        initial_means (list, optional): Initial cluster centroids. If None, selected randomly
        verbose (bool): Whether to print iteration progress (default: False)
        return_distances (bool): Whether to return distance matrix (default: False)
        seed (int): Random seed for initial centroid selection (default: 2026)
    
    Returns:
        tuple: If return_distances is False: (means, indexings, matching)
            - means (np.ndarray): Array of K cluster centroids, each with k district vectors
            - indexings (list): List of district alignments for each plan
            - matching (list): Cluster assignment for each plan
            
            If return_distances is True, also includes:
            - family_to_mean_distances (np.ndarray): Distance matrix of shape (num_families, K)
    
    Raises:
        Warning: If some clusters have no families assigned after convergence
    """
    all_sq_distances = None
    if num_precincts is None:
        num_precincts = len(families[0][0])
    k = len(families[0])

    def get_indexing(plan_idx, mean_idx):
        M = all_sq_distances[mean_idx*k:(mean_idx+1)*k,:][:,plan_idx*k:(plan_idx+1)*k]
        row_ind, col_ind = LSA(M)
        return col_ind, sum(M[r,c] for r, c in zip(row_ind, col_ind))**(1/2)

    if initial_means is not None:
        means = initial_means.copy()
    else:
        np.random.seed(seed)
        mean_idxs = np.random.choice(len(families), K, replace=False)
        means = [families[i].copy() for i in mean_idxs]
    unrolled_families = np.array([d for p in families for d in p])
    families_norms = np.array(np.linalg.norm(unrolled_families, axis=1), dtype=np.float64)**2

    matching = [0]*len(families)
    for iter in range(max_iter):
        unrolled_means = np.array([d for p in means for d in p])
        if verbose:
            print("distance calculation", end=" | ")
        all_sq_distances = euclidean_distances(unrolled_means, unrolled_families, Y_norm_squared=families_norms)**2
        if verbose:
            print("aligning", end=" | ")
        indexings_and_distances = [
            [get_indexing(i, j) for i, p in enumerate(families)] for j, mean in enumerate(means)
        ]
        if verbose:
            print("matching", end=" | ")
        family_to_mean_distances = np.array([
            [indexings_and_distances[j][i][1] for j in range(K)] for i, p in enumerate(families)
        ])
        matching = [np.argmin(family_to_mean_distances[i]) for i, p in enumerate(families)]
        indexings = [indexings_and_distances[matching[i]][i][0] for i, p in enumerate(families)]
        if verbose:
            print('D={:e}'.format(sum(
                indexings_and_distances[matching[i]][i][1]**q for i, p in enumerate(families)
            )**(1/q)), end=" | ")
        if verbose:
            print("recalculating means", end=" | ")
        newmeans = []
        for m, mean in enumerate(means):
            newmeans.append([
                get_mean_vectors([p[indexings[j][i]] for j, p in enumerate(families) if matching[j]==m], num_precincts) for i in range(len(mean))
            ])
        change = max(distance_function(mean[i], newmean[i]) for mean, newmean in zip(means, newmeans) for i in range(len(mean)))
        if verbose:
            print("iter {}: change to barycenter = {:.2f}".format(iter, change))
        if change <= eps:
            if set(matching) != set(range(K)) and verbose:
                print('Warning: some means have no families assigned to them.')
            if not return_distances:
                return np.array(newmeans), indexings, matching
            else:
                return np.array(newmeans), indexings, matching, family_to_mean_distances
        else:
            means = newmeans.copy()
    print('Did not converge')
    if not return_distances:
        return np.array(newmeans), indexings, matching
    else:
        return np.array(newmeans), indexings, matching, family_to_mean_distances

def k_means_distortion(families, newmeans, indexings, matching, q=2):
    """Calculate total distortion (q-norm to the power q) for k-means clustering.
    
    Distortion measures the sum of distances raised to the qth power between each
    plan and its assigned cluster centroid, after optimal district alignment.
    
    Args:
        families (list): List of redistricting plans
        newmeans (list): List of cluster centroids
        indexings (list): List of district alignments for each family
        matching (list): Cluster assignment for each family (indices into newmeans)
        q (int): Norm parameter for distance calculation (default: 2)
    
    Returns:
        float: Total distortion across all plans
    """
    distortions = []
    for i, family in enumerate(families):
        cluster = matching[i]
        indexing = indexings[i]
        mean = newmeans[cluster]
        distortion = sum(distance_function(family[indexing[j]], mean[j], ord=q)**q for j in range(len(family)))
        distortions.append(distortion)
    return np.sum(distortions)


def plot_centroids_as_icecream(centroids, coords=None, pctSF=None, cntySF=None, colors=None, ax=None):
    """Plot cluster centroids with graduated transparency ('icecream' visualization).
    
    This visualization style shows district membership probabilities using a gradient
    of transparency, where higher membership values are more opaque. It creates an
    "icecream cone" effect when multiple districts overlap.
    
    Args:
        centroids (list): List of k cluster centroid vectors, each of length num_precincts
        coords (np.ndarray, optional): Precinct coordinates of shape (num_precincts, 2) for scatter plot
        pctSF (gpd.GeoDataFrame, optional): Precinct shapefile for choropleth visualization
        cntySF (gpd.GeoDataFrame, optional): County shapefile for boundary overlay
        colors (list, optional): Color indices for each district. If None, uses sequential integers
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, uses current axes
    
    Raises:
        ValueError: If neither pctSF nor coords is provided
    """
    if ax is None:
        ax = plt.gca()
    if cntySF is not None:
        cntySF.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
    k = len(centroids)
    if colors is None:
        colors = list(range(k))
    for d in range(k):
        mycolorarray = np.array([mycmap_alpha(a,d, colors=colors) for a in np.linspace(0,1,10)])
        newcmp = ListedColormap(mycolorarray)
        in_district = centroids[d]
        if pctSF is not None:
            pctSF['in'] = in_district
            pctSF.plot(column='in',cmap=newcmp, ax=ax, edgecolor='none')
        elif coords is not None:
            ax.scatter(coords[:,0], coords[:,1], alpha=in_district/max(in_district), color=mycmap(colors[d]), s=25, marker='s')
        else:
            raise ValueError("Either pctSF or coords must be provided for visualization")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)