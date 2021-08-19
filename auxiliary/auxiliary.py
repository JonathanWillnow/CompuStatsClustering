import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as tools
import scipy.stats as scipy
import sklearn.datasets as sk_data
import sklearn.cluster as sk_cluster
import sklearn_extra.cluster as skx_cluster
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import random
import kneed


import time


def generate_blobs(n_samples, centers, cluster_std, random_state):
    features, true_labels = sk_data.make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

    # Standartization
    scaler = sk_preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)
    blobs_df = pd.DataFrame(features_scaled, columns = ["x1", "x2"])
    blobs_df["label"] = true_labels

    return(blobs_df)



def generate_spiraldata(n, plotting):
    
    n_samples = n
    t = 1.25 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    X = np.concatenate((x, y))
    X += .7 * np.random.randn(2, n_samples)
    X = X.T
    
    if plotting == True:
        fig, ax = plt.subplots(figsize = (10,10))
        ax = sns.scatterplot(X[:, 0], X[:, 1],cmap=plt.cm.nipy_spectral)
        ax.set_title("complex dataset")
        
    return(X)

###############################################################################

def plot_generated_datasets(complex_data,blobs_df):

    fig, ax = plt.subplots(1,2,figsize = (22,10))
    plt.subplot(121)
    ax = sns.scatterplot(x= blobs_df.x1, y=blobs_df.x2, hue=blobs_df.label)
    ax.set_title("Isotropic dataset");

    plt.subplot(122)
    ax = sns.scatterplot(x= complex_data[:, 0],  y= complex_data[:, 1],cmap=plt.cm.nipy_spectral)
    ax.set_title("Complex spiral dataset")


    
    
    ### https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
def plot_clusters(data, data_c, algorithm, args, kwds):
        
        
    fig, ax = plt.subplots(1,2, figsize = (20,10))
    
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data[["x1", "x2"]])
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    #colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    
    results = pd.DataFrame(data, columns = ["x1", "x2"])
    results["labels"] = labels
    
    plt.subplot(121)
    ax = sns.scatterplot(x=results.x1, y=results.x2, hue = labels)
    ax.set_title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
#########
    start_time = time.time()
    labels_c = algorithm(*args, **kwds).fit_predict(data_c)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    #colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    
    results_c = pd.DataFrame(data_c, columns = ["x1", "x2"])
    results_c["labels_c"] = labels_c
    
    plt.subplot(122)
    ax = sns.scatterplot(x=results_c.x1, y=results_c.x2, hue = labels_c)
    ax.set_title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    
    
    
    
    return(labels, labels_c)


####################################################################





def plot_Agglomerative_clusters(data,data_c, n, args):
    
    
    kwds = {'n_clusters':n, 'linkage':'ward'}
    fig, ax = plt.subplots(4,2, figsize = (20,40))
  
    start_time = time.time()
    labels = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data)
    end_time = time.time()
    
    results = pd.DataFrame(data, columns = ["x1", "x2"])
    results["labels"] = labels
    
    plt.subplot(421)
    ax = sns.scatterplot(x=results.x1, y=results.x2, hue = labels)
    ax.set_title('Clusters found using ward' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
##
    kwds = {'n_clusters':n, 'linkage':'ward'}
  
    start_time = time.time()
    labels_c = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data_c)
    end_time = time.time()
    
    results_c = pd.DataFrame(data_c, columns = ["x1", "x2"])
    results_c["labels_c"] = labels_c
    
    plt.subplot(422)
    ax = sns.scatterplot(x=results_c.x1, y=results_c.x2, hue = labels_c)
    ax.set_title('Clusters found using ward' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

###########
    kwds = {'n_clusters':n, 'linkage':'complete'}
    start_time = time.time()
    labels = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data)
    end_time = time.time()
    
    results = pd.DataFrame(data, columns = ["x1", "x2"])
    results["labels"] = labels
    
    plt.subplot(423)
    ax = sns.scatterplot(x=results.x1, y=results.x2, hue = labels)
    ax.set_title('Clusters found using complete linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
##
    kwds = {'n_clusters':n, 'linkage':'complete'}
  
    start_time = time.time()
    labels_c = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data_c)
    end_time = time.time()
    
    results_c = pd.DataFrame(data_c, columns = ["x1", "x2"])
    results_c["labels_c"] = labels_c
    
    plt.subplot(424)
    ax = sns.scatterplot(x=results_c.x1, y=results_c.x2, hue = labels_c)
    ax.set_title('Clusters found using complete linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
###########
    kwds = {'n_clusters':n, 'linkage':'average'}
    start_time = time.time()
    labels = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data)
    end_time = time.time()
    
    results = pd.DataFrame(data, columns = ["x1", "x2"])
    results["labels"] = labels
    
    plt.subplot(425)
    ax = sns.scatterplot(x=results.x1, y=results.x2, hue = labels)
    ax.set_title('Clusters found using average linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
##
    kwds = {'n_clusters':n, 'linkage':'average'}
  
    start_time = time.time()
    labels_c = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data_c)
    end_time = time.time()
    
    results_c = pd.DataFrame(data_c, columns = ["x1", "x2"])
    results_c["labels_c"] = labels_c
    
    plt.subplot(426)
    ax = sns.scatterplot(x=results_c.x1, y=results_c.x2, hue = labels_c)
    ax.set_title('Clusters found using average linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
###########

    kwds = {'n_clusters':n, 'linkage':'single'}
    start_time = time.time()
    labels = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data)
    end_time = time.time()
    
    results = pd.DataFrame(data, columns = ["x1", "x2"])
    results["labels"] = labels
    
    plt.subplot(427)
    ax = sns.scatterplot(x=results.x1, y=results.x2, hue = labels)
    ax.set_title('Clusters found using single linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
##
    kwds = {'n_clusters':n, 'linkage':'single'}
  
    start_time = time.time()
    labels_c = sk_cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data_c)
    end_time = time.time()
    
    results_c = pd.DataFrame(data_c, columns = ["x1", "x2"])
    results_c["labels_c"] = labels_c
    
    plt.subplot(428)
    ax = sns.scatterplot(x=results_c.x1, y=results_c.x2, hue = labels_c)
    ax.set_title('Clusters found using single linkage' , fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
###########


#############################################################################




def kmeans_validation_example(kmeans_kwargs, kmax, data):
    
    sse = []
    for k in range(1, kmax):
        kmeans = sk_cluster.KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data[["x1", "x2"]])
        sse.append(kmeans.inertia_)


    kl = kneed.KneeLocator(range(1, kmax), sse, curve="convex", direction="decreasing")
   # kl.elbow
    
    silhouette_coef = []
    for k in range(2, kmax):
        kmeans = sk_cluster.KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data[["x1", "x2"]])
        score = round(sk_metrics.silhouette_score(data, kmeans.labels_),4)
        silhouette_coef.append(score)
        
    max_score = max(silhouette_coef)
    best_k = 2+silhouette_coef.index(max_score)

        
    
    fig, ax = plt.subplots(2,2, figsize = (20,20))
    
   
    plt.subplot(221)
    ax = sns.scatterplot(x= data.x1, y=data.x2, hue=data.label)
    #ax.set_xlabel("number of clusters")
    ax.set_title("Original dataset");    
    
    
    
    labels_optimal = sk_cluster.KMeans(n_clusters=best_k, **kmeans_kwargs).fit_predict(data[["x1", "x2"]])
    results_optimal = pd.DataFrame(data, columns = ["x1", "x2"])
    results_optimal["labels"] = labels_optimal

    

    plt.subplot(222)
    ax = sns.scatterplot(x= results_optimal.x1, y=results_optimal.x2, hue=results_optimal.labels)
    ax.set_title("Predicted clusters");
    #fig.suptitle(f"k-means, Specifications: Centers: {centers}, Cluster std.: {cluster_std}", fontsize=16)
    
    plt.subplot(223)
    ax = sns.lineplot(x= range(1,kmax),y=sse)
    ax1 = plt.axvline(x=kl.elbow, animated = True, ls = "--", c = "red")
    ax.set_ylabel("SSE")
    ax.set_xlabel("number of clusters")
    ax.set_title("Ellbow Method");    

    plt.subplot(224)
    ax = sns.lineplot(x=range(2,kmax), y=silhouette_coef)
    ax1 = plt.axvline(x=best_k, animated = True, ls = "--", c = "red")
    ax.set_ylabel("SSE")
    ax.set_xlabel("number of clusters")
    ax.set_title("Silhouette Coefficient");
    fig.suptitle("K-Means", fontsize=20)
    plt.tight_layout(pad=2.5);


    #####################################################################################
    
def kmedoids_validation_example(kmedoids_kwargs, kmax, data):
    
    sse = []
    for k in range(1, kmax):
        kmedoids = skx_cluster.KMedoids(n_clusters=k, **kmedoids_kwargs )
        kmedoids.fit(data[["x1", "x2"]])
        sse.append(kmedoids.inertia_)


    kl = kneed.KneeLocator(range(1, kmax), sse, curve="convex", direction="decreasing")
    kl.elbow
    
    silhouette_coef = []
    for k in range(2, kmax):
        kmedoids = skx_cluster.KMedoids(n_clusters=k, **kmedoids_kwargs )
        kmedoids.fit(data[["x1", "x2"]])
        score = round(sk_metrics.silhouette_score(data, kmedoids.labels_),4)
        silhouette_coef.append(score)
        
    max_score = max(silhouette_coef)
    best_k = 2+silhouette_coef.index(max_score)

        
    
    fig, ax = plt.subplots(2,2, figsize = (20,20))
    
   
    plt.subplot(221)
    ax = sns.scatterplot(x= data.x1, y=data.x2, hue=data.label)
    #ax.set_xlabel("number of clusters")
    ax.set_title("Original dataset");    
    
    
    
    labels_optimal = skx_cluster.KMedoids(n_clusters=best_k, **kmedoids_kwargs ).fit_predict(data[["x1", "x2"]])
    results_optimal = pd.DataFrame(data, columns = ["x1", "x2"])
    results_optimal["labels"] = labels_optimal

    

    plt.subplot(222)
    ax = sns.scatterplot(x= results_optimal.x1, y=results_optimal.x2, hue=results_optimal.labels)
    #ax1 = plt.axvline(x=best_k, animated = True, ls = "--", c = "red")
    ax.set_title("Predicted clusters");
    #fig.suptitle(f"k-means, Specifications: Centers: {centers}, Cluster std.: {cluster_std}", fontsize=16)
    
    plt.subplot(223)
    ax = sns.lineplot(x= range(1,kmax),y=sse)
    ax1 = plt.axvline(x=kl.elbow, animated = True, ls = "--", c = "red")
    ax.set_ylabel("SSE")
    ax.set_xlabel("number of clusters")
    ax.set_title("Ellbow Method");    

    plt.subplot(224)
    ax = sns.lineplot(x=range(2,kmax), y=silhouette_coef)
    ax1 = plt.axvline(x=best_k, animated = True, ls = "--", c = "red")
    ax.set_ylabel("SSE")
    ax.set_xlabel("number of clusters")
    ax.set_title("Silhouette Coefficient");
    fig.suptitle("K-Medoids", fontsize=20)
    plt.tight_layout(pad=2.5);




#########################################################################    



def benchmark_algorithm(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=2):

    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = pd.DataFrame(np.nan * np.ones((len(dataset_sizes), sample_size)), columns = ["nobs","time"])
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sk_data.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters)

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time
            
            # create list to temporarily store results
            h_result = []
            h_result.append(time_taken)
            
        # calculate mean of time taken and add to result DataFRame
        result.loc[index, "time"] = (sum(h_result)/len(h_result))
        result.loc[index, "nobs"] = size

    # Return the result as a dataframe for easier handling with seaborn afterwards
    #return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                  # result.flatten()]).T, columns=['x','y'])
    return(result)



#########################################################################



def benchmark(kwargs_random, kwargs_affin):
    
    dataset_sizes = np.hstack([np.arange(1, 4) * 500])#, np.arange(3,7) * 1000, np.arange(4,10) * 2000])
     
#########
    k_means = sk_cluster.KMeans(10, **kwargs_random)
    k_means_data = benchmark_algorithm(dataset_sizes, k_means.fit, (), {})

    k_medoids = skx_cluster.KMedoids(**kwargs_random)
    k_medoids_data = benchmark_algorithm(dataset_sizes, k_medoids.fit, (), {})
    #{'n_clusters':3,  "init": "random", "max_iter": 300, "random_state": 42})


    mean_shift = sk_cluster.MeanShift(10)
    mean_shift_data = benchmark_algorithm(dataset_sizes, mean_shift.fit, (), {})

    affinity_propagation = sk_cluster.AffinityPropagation(**kwargs_affin);
    affinity_propagation_data = benchmark_algorithm(dataset_sizes, affinity_propagation.fit, (), {});

    agglomarative_clustering = sk_cluster.AgglomerativeClustering();
    agglomarative_clustering_data = benchmark_algorithm(dataset_sizes, agglomarative_clustering.fit, (), {});

    dbscan = sk_cluster.DBSCAN()
    dbscan_data = benchmark_algorithm(dataset_sizes, dbscan.fit, (), {})

##########

    fig,ax = plt.subplots(figsize = (10,10))
    ax = sns.lineplot(x= 'nobs', y='time', data=k_means_data, label='Sklearn K-Means')
    ax = sns.lineplot(x= 'nobs', y='time', data=k_medoids_data, label='Sklearn K-Medoids')
    ax = sns.lineplot(x= 'nobs', y='time', data=mean_shift_data, label='Sklearn Meanshift')
    ax = sns.lineplot(x= 'nobs', y='time', data=affinity_propagation_data, label='Sklearn Affinity Propagation')
    ax = sns.lineplot(x= 'nobs', y='time', data=agglomarative_clustering_data, label='Sklearn Agglomerative Clustering')
    ax = sns.lineplot(x= 'nobs', y='time', data=dbscan_data, label='Sklearn DBSCAN')
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Time Taken")
    plt.plot();

#########################################################################

    