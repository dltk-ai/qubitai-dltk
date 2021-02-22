Clustering
===========

*Supported Libraries and Algorithms*

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - scikit
     - weka
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

    :param service: clustering
    :param algorithm: algorithm to use for clustering
    :param dataset: dataset file location in DLTK storage.
    :param features: List of features to use for clustering.
    :param lib: Library for clustering the model.
    :param number_of_clusters: number of clusters to divide the data into.
    :param model_name: Model will be saved with the name specified in this parameter.
    :param save_model: If True, the model will be saved in DLTK Storage.
    :param params: additional parameters
    :param dataset_source: To specify data source,
        None: Dataset file will from DLTK storage will be used
        database: Query from connected database will be used
    :rtype: A json obj containing model info.

example::

    import dltk_ai
    from dltk_ai.dataset_types import Dataset
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    task = "cluster"
    library = "scikit"
    algorithm = "KMeansClustering"
    features = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    label = 'MEDV'
    train_percentage = 80
    model_name = "HousePricePrediction"
    save_model = "true"

    train_response = client.train(task, algorithm, train_data, label,features,model_name, library, train_percentage, save_model)
    print(train_response)

