Configurations
===============

Database Configuration
-----------------------

.. glossary::
    a. Postgres

        Postgres is used for saving output of various services. We can deploy postgres or use an existing postgres DB.

        +-------------------+-----------------------------------------------+-------------------+
        | Parameter         |                Description                    |  Default          |
        +===================+===============================================+===================+
        | *SQL_USER*        | User Name for postgres database               |  postgres         |
        +-------------------+-----------------------------------------------+-------------------+
        | *SQL_PASSWORD*    | Password for postgres database                |  postgres         |
        +-------------------+-----------------------------------------------+-------------------+
        | *SQL_HOST*        | IP address of the postgres database           |  http://localhost |
        +-------------------+-----------------------------------------------+-------------------+
        | *SQL_PORT*        | Port where postgres database can be accessed  |  5432             |
        +-------------------+-----------------------------------------------+-------------------+


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
        | *REDIS_IP*        | User Name for postgres database               |
        +-------------------+-----------------------------------------------+
        | *REDIS_PORT*      | Password for postgres database                |
        +-------------------+-----------------------------------------------+

        .. code-block::

            # Redis Details
            REDIS_IP="12.1.1.8"
            REDIS_PORT="6379"

    c. InfluxDB

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
            REGISTRY_SERVICE_IP="http://12.1.1.9"
            REGISTRY_SERVICE_PORT="8761"


    b. Kong

        Kong is used for ....


    c. Solution Service

        .. code-block::

            SOLUTION_INSTANCE_IP="dltk-solution-service"
            SOLUTION_INSTANCE_PORT="8093"


Storage Configurations
------------------------

    .. glossary::
        a. Local

            .. code-block::

                STORAGE_TYPE="local"

        b. AWS S3

            .. code-block::

                # S3 Config
                S3_BUCKET="YOUR_S3_BUCKET"
                S3_REGION="ap-south-1"
                S3_ENDPOINT="YOUR_S3_ENDPOINT"
                S3_ACCESS_KEY="YOUR_S3_ACCESS_KEY"
                S3_SECRET_KEY="YOUR_S3_SECRET_KEY"