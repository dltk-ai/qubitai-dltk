import dltk_ai
from dltk_ai.dataset_types import Dataset


def main():
    c = dltk_ai.DltkAiClient('YOUR_APIKEY')

    cluster_data_store_response = c.store('../csv/moon_data.csv', Dataset.TRAIN_DATA)
    print(cluster_data_store_response)
    cluster_data = cluster_data_store_response['fileUrl']

    cluster_response = c.cluster("clustering", "KMeansClustering", cluster_data, ["X", "Y"], "scikit", 2,"Clustering_Model", True,None)
    print(cluster_response)
    cluster_job_status_response = c.job_status(cluster_response['data']['jobId'])
    print(cluster_job_status_response)
    cluster_job_output_response = c.job_output(cluster_response['data']['jobId'])
    print(cluster_job_output_response)

    pred_file = cluster_job_output_response['output']['clusterFileUrl']
    response = c.download(pred_file)
    print(response.text)


if __name__ == '__main__':
    main()
