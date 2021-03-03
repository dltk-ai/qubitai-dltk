Clustering
===========

Clustering is a technique used in grouping data points into desired number of groups (clusters), based on similarity of data points.

*Supported Libraries and Algorithms*

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Scikit
     - Weka
   * - AgglomerativeClustering
     - SimpleKMeans
   * - KMeansClustering 
     - 
   * - DBScan
     - 
   * - MiniBatchKMeans
     - 
   * - Birch
     - 
   * - SpectralClustering
     - 
    
Cluster
-------

.. function:: client.cluster(service, algorithm, dataset, features, lib='weka'
                            , number_of_clusters=2, model_name=None,
                            save_model=True, params=None, dataset_source=None):

    :param service: Clustering
    :param algorithm: Algorithm to use for clustering
    :param dataset: Dataset file location in DLTK storage.
    :param features: List of features to use for clustering.
    :param lib: Library for clustering the model.
    :param number_of_clusters: Number of clusters to divide the data into.
    :param model_name: Model will be saved with the name specified in this parameter.
    :param save_model: If True, the model will be saved in DLTK Storage.
    :param params: Additional parameters
    :param dataset_source: To specify data source,
        None: Dataset file will from DLTK storage will be used
        database: Query from connected database will be used
    :rtype: A json object containing model info.


