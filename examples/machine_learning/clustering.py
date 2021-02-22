import dltk_ai
from dltk_ai.dataset_types import Dataset

client = dltk_ai.DltkAiClient('YOUR_APIKEY')

cluster_data_store_response = client.store('../examples/data/csv/moon_data.csv', Dataset.TRAIN_DATA)
print(cluster_data_store_response)
cluster_data = cluster_data_store_response['fileUrl']

# clustering
# params
task = "clustering"
algorithm = "KMeansClustering"
features = ["X", "Y"]
library = "scikit"
number_of_clusters = 2000
save_model = True

cluster_response = client.cluster(task, algorithm, cluster_data, features, library, number_of_clusters,
                                  "Clustering_Model", save_model)
print(cluster_response)

# job status of clustering
cluster_job_status_response = client.job_status(cluster_response['data']['jobId'])
print(cluster_job_status_response)

# output response of clustering job
cluster_job_output_response = client.job_output(cluster_response['data']['jobId'])
print(cluster_job_output_response)

# download the predictions file which contains the clusters
pred_file = cluster_job_output_response['output']['clusterFileUrl']
response = client.download(pred_file)
print(response.text)
