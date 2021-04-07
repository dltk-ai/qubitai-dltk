Configurations
===============

Database Configuration
-----------------------

    .. glossary::
        a. Postgres

            Postgres is used for saving output of various services. We can deploy postgres or use an existing postgres DB.

            +-------------------+-----------------------------------------------+
            | Parameter         |                Description                    |
            +===================+===============================================+
            | *SQL_USER*        | User Name for postgres database               |
            +-------------------+-----------------------------------------------+
            | *SQL_PASSWORD*    | Password for postgres database                |
            +-------------------+-----------------------------------------------+
            | *SQL_HOST*        | IP address of the postgres database           |
            +-------------------+-----------------------------------------------+
            | *SQL_PORT*        | Port where postgres database can be accessed  |
            +-------------------+-----------------------------------------------+


            .. code-block::

                # postgres
                PYTHON_SQL_ENGINE="django.db.backends.postgresql"
                SQL_USER="postgres"
                SQL_PASSWORD="postgres"
                SQL_HOST="12.1.1.8"
                SQL_PORT="5432"
                EXTERNAL_PORT="5432"
                JAVA_SQL_DRIVER="org.postgresql.Driver"

        b. Redis

            +-------------------+-----------------------------------------------+
            | Parameter         |                Description                    |
            +-------------------+-----------------------------------------------+
            | *REDIS_IP*        | User Name for postgres database               |
            +-------------------+-----------------------------------------------+
            | *REDIS_PORT*      | Password for postgres database                |
            +-------------------+-----------------------------------------------+

            .. code-block::

                # Redis Details
                REDIS_IP="12.1.1.8"
                REDIS_PORT="6379"

        c. InfluxDB

            +-------------------------+------------------------------------------------+
            | Parameter               |           Description                          |
            +=========================+================================================+
            | *INFLUXDB_IP*           |  IP address of the postgres database           |
            +-------------------------+------------------------------------------------+
            | *INFLUXDB_PORT*         |  Port where postgres database can be accessed  |
            +-------------------------+------------------------------------------------+
            | *INFLUXDB_USER*         |  User Name for influx database                 |
            +-------------------------+------------------------------------------------+
            | *INFLUXDB_PASSWORD*     |  Password for influx database                  |
            +-------------------------+------------------------------------------------+

            .. code-block::

                # InfluxDb Details
                INFLUXDB_IP="12.1.1.8"
                INFLUXDB_PORT="8011"
                INFLUXDB_USER="influxdb"
                INFLUXDB_PASSWORD="influxdb"

Base Services
-------------

    .. glossary::
        a. Registry Service

            .. code-block::

                # Registry service Details
                REGISTRY_SERVICE_IP="http://YOUR_BASE_SERVICE_IP_ADDRESS"
                REGISTRY_SERVICE_PORT="8761"


        b. Solution Service

            .. code-block::

                SOLUTION_INSTANCE_IP="YOUR_BASE_SERVICE_IP_ADDRESS"
                SOLUTION_INSTANCE_PORT="8093"


Storage Configurations
------------------------

    .. glossary::
        a. Local

            .. code-block::

                STORAGE_TYPE="local"

        b. AWS S3

            - Refer this `link <https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html>`__ for creating a bucket in AWS S3.

            .. code-block::

                STORAGE_TYPE="aws"

                # S3 Config
                S3_BUCKET="YOUR_S3_BUCKET"
                S3_REGION="ap-south-1"
                S3_ENDPOINT="YOUR_S3_ENDPOINT"
                S3_ACCESS_KEY="YOUR_S3_ACCESS_KEY"
                S3_SECRET_KEY="YOUR_S3_SECRET_KEY"

        b. GCP

            - GCP_SERVICE_ACCOUNT_FILE will contain your GCS credentials.You can generate this by following this `link <https://cloud.google.com/iam/docs/creating-managing-service-accounts>`__

            .. code-block::

                STORAGE_TYPE="gcp"

                # GCP
                GCP_SERVICE_ACCOUNT_FILE_PATH="YOUR_GCP_SERVICE_ACCOUNT_PATH"
                GCP_PRIVATE_BUCKET="YOUR_GCP_PRIVATE_BUCKET"
                GCP_PUBLIC_BUCKET="YOUR_GCP_PUBLIC_BUCKET"

        b. DIGITAL OCEAN

            - Refer this `link <https://www.digitalocean.com/docs/spaces/how-to/create/>`__ for creating a bucket in Digital Ocean Spaces.

            .. code-block::

                STORAGE_TYPE="do"

                # Digital Ocean
                DO_ENDPOINT="YOUR_DO_ENDPOINT"
                DO_ACCESS_KEY="YOUR_DO_ACCESS_KEY"
                DO_SECRET_KEY="YOUR_DO_SECRET_KEY"
                DO_BUCKET="YOUR_DO_BUCKET"
                DO_REGION="YOUR_DO_REGION"

Authentication Configurations
-----------------------------

    .. glossary::

        a. Enable Auth

            .. code-block::

                AUTH_ENABLED="true"
                SMTP_HOST="YOUR_SMTP_HOST"
                SMTP_PORT="587"
                SMTP_USERNAME="YOUR_SMTP_USERNAME"
                SMTP_PASSWORD="YOUR_SMTP_PASSWORD"
                UI_SERVICE_URL="http://YOUR_UI_SERVICE_IP_ADDRESS:8082"

        b. Disable Auth

            .. code-block::

                AUTH_ENABLED="false"


AI Engines Credentials
----------------------


    - Refer this `link <https://ms.portal.azure.com/#create/Microsoft.CognitiveServicesTextAnalytics>`__ to create Azure Language Subscription Keys.
    - Refer this `link <https://ms.portal.azure.com/#create/Microsoft.CognitiveServicesComputerVision>`__ to create Azure Vision Subscription Keys.
    - Refer this `link <https://ms.portal.azure.com/#create/Microsoft.CognitiveServicesFace>`__ to create Azure Face Subscription Keys.
    - Refer this `link <https://www.ibm.com/watson/developercloud/doc/virtual-agent/api-keys.html>`__ to create IBM Watson API Keys.

    .. glossary::

        a. NLP

            .. code-block::

                # URL is given only for reference, replace with your credentials

                AZURE_LANGUAGE_APIKEY="USER_DEFINED"
                AZURE_LANGUAGE_URL="https://dltk-text-analytics.cognitiveservices.azure.com/text/analytics/v2.1/"

                IBM_LANGUAGE_URL="https://gateway-lon.watsonplatform.net/natural-language-understanding/api"
                IBM_LANGUAGE_APIKEY="USER_DEFINED"

        b. Computer Vision
            For Object Detection and Image Classification

            .. code-block::

                # URL is given only for reference, replace with your credentials

                AZURE_VISION_SUBSCRIPTION_KEY="USER_DEFINED"
                AZURE_VISION_URL="https://dltk-ai-cv.cognitiveservices.azure.com/vision/v2.0/analyze"

                IBM_VISUAL_URL="https://gateway.watsonplatform.net/visual-recognition/api"
                IBM_VISUAL_APIKEY="USER_DEFINED"

        b. Face Analytics

            .. code-block::

                # URL is given only for reference, replace with your credentials

                AZURE_FACE_DETECTION_URL="https://dltk-ai-face.cognitiveservices.azure.com/face/v1.0/detect"
                AZURE_FACE_DETECTION_SUBSCRIPTION_KEY ="USER_DEFINED"


Microservices Instance IP and Ports
-----------------------------------

    .. glossary::

        a. ML WRAPPER SERVICE

            .. code-block::

                ML_WRAPPER_INSTANCE_IP="YOUR_DLTK_AI_ML_WRAPPER_HOST_IP_ADDRESS"
                ML_WRAPPER_INSTANCE_PORT="8199"

        b. ML WEKA SERVICE

            .. code-block::

                ML_WEKA_INSTANCE_IP="YOUR_DLTK_AI_ML_WRAPPER_HOST_IP_ADDRESS"
                ML_WEKA_INSTANCE_PORT="8101"

        c. ML SCIKIT SERVICE

            .. code-block::

                ML_SCIKIT_INSTANCE_IP="YOUR_DLTK_AI_ML_WRAPPER_HOST_IP_ADDRESS"
                ML_SCIKIT_INSTANCE_PORT="8089"

        d. ML H2O SERVICE

            .. code-block::

                ML_H2O_INSTANCE_IP="YOUR_DLTK_AI_ML_WRAPPER_HOST_IP_ADDRESS"
                ML_H2O_INSTANCE_PORT="8087"

        e. NLP SERVICE

            .. code-block::

                NLP_INSTANCE_IP="YOUR_DLTK_AI_WRAPPER_HOST_IP_ADDRESS"
                NLP_INSTANCE_PORT="8189"

        f. CV WRAPPER SERVICE

            .. code-block::

                CV_WRAPPER_INSTANCE_IP="YOUR_DLTK_AI_WRAPPER_HOST_IP_ADDRESS"
                CV_WRAPPER_INSTANCE_PORT="8190"
