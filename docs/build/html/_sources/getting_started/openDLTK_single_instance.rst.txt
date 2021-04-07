*****************************
OpenDLTK on single machine
*****************************

.. contents:: Table of Contents
    :depth: 4
    :local:

Pre-requisites
================


.. tab:: Linux

    **1. Python 3.6+** : To install python refer this `Python installation guide <https://realpython.com/installing-python/>`__

    **2. Virtual Environment** : To install virtual environment refer this `Virtual Environment installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`__

    **3. Docker** : To install docker refer this `Docker installation guide <https://docs.docker.com/engine/install/>`__



.. tab:: Mac

    **1. Python 3.6+** : To install python refer this `Python installation guide <https://realpython.com/installing-python/>`__

    **2. Virtual Environment** : To install virtual environment refer this `Virtual Environment installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`__

    **3. Docker** : To install docker refer this `Docker installation guide <https://docs.docker.com/docker-for-mac/install/>`__



.. tab:: Windows

    **1. Python 3.6+** : To install python refer this `Python installation guide <https://realpython.com/installing-python/>`__

    **2. Virtual Environment** : To install virtual environment refer this `Virtual Environment installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`__

    **3. Docker** : To install docker refer this `Docker installation guide <https://docs.docker.com/docker-for-windows/install-windows-home/>`__

.. warning::

    As these deep learning models are computationally expensive, its recommended to run these docker containers on minimum of 16GB RAM machine.

Installation
=============

**1. Clone OpenDLTK github repository**

.. tab:: Linux

    .. code-block::

        git clone https://github.com/dltk-ai/OpenDLTK
        cd OpenDLTK
        pip install -r requirements.txt
        sudo groupadd docker
        sudo usermod -aG docker $(whoami)

    Log out and Log back in to ensure docker runs with correct permissions. (if on remote machine,please reboot the server)

    .. code-block::

        sudo service docker start

.. tab:: Mac

    .. code-block::

        git clone https://github.com/dltk-ai/OpenDLTK
        cd OpenDLTK
        pip install -r requirements.txt

.. tab:: Windows

    .. code-block::

        git clone https://github.com/dltk-ai/OpenDLTK
        cd OpenDLTK
        sudo pip install -r requirements.txt

**2. Initialize DLTK setup**

`cd` to directory containing `setup.py` file and use below command to start installation process.

.. tab:: Linux

    .. code-block::

        sudo python setup.py -m init

    .. code-block::

       Which version you want to install ['1.0', '1.1']
       Enter your input: 1.0


.. tab:: Mac

    .. code-block::

        sudo python setup.py -m init

.. tab:: Windows

    .. code-block::

        python setup.py -m init

You can choose version of openDLTK you want to install on your machine.
Please ensure this version should be compatible with the `Python client SDK <https://github.com/dltk-ai/qubitai-dltk>`__ you installed above.

**3. Updating Configuration**

.. tab:: Linux

    Please update config.env file saved at ``/usr/dltk-ai/config.env``

.. tab:: Mac

    Please update config.env file saved at ``/usr/dltk-ai/config.env``

.. tab:: Windows

    Please update **config.env** file saved at ``C:\Users\{username}\AppData\Local\dltk-ai\config.env``

|

*a. Configuring Storage*

    .. tab:: Local

        .. code-block::

            STORAGE_TYPE="local"

    .. tab:: AWS S3

        .. code-block::

            STORAGE_TYPE="aws"

            # Values only for reference, replace with your credentials

            S3_ACCESS_KEY="AKIAVKNVW3O4G2YSG"
            S3_SECRET_KEY="vrJvyZFGSpOFTtZcsDTZTHwJ88Jw"
            S3_BUCKET="dltk-ai"
            S3_REGION="ap-south-1"
            S3_ENDPOINT="https://s3.ap-south-1.amazonaws.com"

        Refer this `link <https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html>`__ for creating a bucket in AWS S3.

    .. tab:: Google Cloud Storage

        .. code-block::

            STORAGE_TYPE="gcp"

            # Values only for reference, replace with your details

            GCP_SERVICE_ACCOUNT_FILE=dltk-ai.json
            GCP_PRIVATE_BUCKET="dltk-ai-private"
            GCP_PUBLIC_BUCKET="dltk-ai-public"

        Replace `base/solution-config/dltk-ai.json <https://github.com/dltk-ai/openDLTK_beta/blob/main/base/solution-config/dltk-ai.json>`__ with your GCS credentials file which you can generate from `GCP service account <https://cloud.google.com/iam/docs/creating-managing-service-accounts>`__

    .. tab:: Digital Ocean

        .. code-block::

            STORAGE_TYPE="do"

            # Values only for reference, replace with your credentials


            DO_ENDPOINT="sgp1.digitaloceanspaces.com"
            DO_ACCESS_KEY="SPZ4OSDVXC35R26"
            DO_SECRET_KEY="9b7SQmnFNx0vzAHWc5czKW75By01CH4"
            DO_BUCKET="dltk-ai"
            DO_REGION="sgp1"

        Refer this `link <https://www.digitalocean.com/docs/spaces/how-to/create/>`__ for creating a bucket in Digital Ocean Spaces.

    .. warning::
        In case you decide to switch your initial storage from one source to another, the data migrations has to be handled by you.


*b. Configure supported AI Engines Credentials*


    .. tab:: Azure

        .. code-block::

            AZURE_LANGUAGE_SUBSCRIPTION_KEY="USER_DEFINED"
            AZURE_BASE_URL="USER_DEFINED"

        .. code-block::

            AZURE_VISION_SUBSCRIPTION_KEY="USER_DEFINED"
            AZURE_VISION_URL="USER_DEFINED"

    .. tab:: IBM

        .. code-block::

            IBM_LANGUAGE_URL="USER_DEFINED"
            IBM_SUBSCRIPTION_KEY="USER_DEFINED"

        .. code-block::

            IBM_VISUAL_URL="USER_DEFINED"
            IBM_VISUAL_APIKEY="USER_DEFINED"

*c. Authentication*

    .. tab:: Enable Authentication

        In config.env file, update

        .. code-block::

            AUTH_ENABLED="true"

            # SMTP setup
            SMTP_HOST="YOUR_SMTP_HOST"
            SMTP_PORT=587
            SMTP_USERNAME="YOUR_SMTP_USERNAME"
            SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

            # UI SERVER URL(replace localhost with server IP in case of remote machine)
            UI_SERVICE_URL="http://localhost:8082"

        .. todo::
            If later you want to disable authentication, please refer this section

    .. tab:: Disable Authentication

        In config.env file, update

        .. code-block::

            AUTH_ENABLED="false"

        .. todo::
            If later you want to enable authentication, please refer this section

**4. Update config**


.. tab:: Linux

    .. code-block::

        sudo python setup.py -m update_config


.. tab:: Mac

    .. code-block::

        sudo python setup.py -m update_config

.. tab:: Windows

    .. code-block::

        python setup.py -m update_config

Result would be like:

.. code-block::

       Which version you want to install ['1.0', '1.1']
       Enter your input: 1.0


**5. Install Services**

.. tab:: Linux

    .. code-block::

        sudo python setup.py -m install


.. tab:: Mac

    .. code-block::

        sudo python setup.py -m install

.. tab:: Windows

    .. code-block::

        python setup.py -m install



You will get a list of service as shown below, choose the services you want to install using comma separated Ids.

::

    Please choose services you want to install from below list
        1. Base
        2. ML Scikit
        3. ML H2O
        4. ML Weka
        5. Image Classification
        6. Object Detection
        7. Face Analytics
        8. Natural Language Processing

    Choose your selection : 5, 8

.. note::

    Image Classification, Object Detection and Face Analytics may take an hour to download.

You can verify whether installation is successful or not by visiting `Registry service <http://localhost:8761>`__ to check status of containers.

Usage
===============

After OpenDLTK is installed, it can be used via DLTK python client SDK, as shown in below example.

.. note::

    Below code block can be run after DLTK python client SDK is installed. The installation of which is covered in next `section <pythonclientsdk.html#installation>`_ .



.. tab:: with Auth Disabled

    .. code-block::

        import dltk_ai

        client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response)

    .. code-block::

            {
              "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
            }


.. tab:: with Auth Enabled

    .. code-block::

        import dltk_ai

        client = dltk_ai.DltkAiClient('86122578-4b01-418d-80cc-049e283d1e2b', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response)

    .. code-block::

        {
          "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
        }


.. seealso::
    1. To enable/disable authentication `link <toggle_auth.html>`__ .
    2. How to Create user and Generate API Key `link <generateAPIkey.html>`__ .

Stop Services
===============

To stop OpenDLTK services, run below commands.

.. tab:: selected services

    .. tab:: Linux

        .. code-block::

            sudo python setup.py --mode uninstall --partial --remove


    .. tab:: Mac

        .. code-block::

            sudo python setup.py --mode uninstall --partial --remove

    .. tab:: Windows

        .. code-block::

            python setup.py --mode uninstall --partial --remove




.. tab:: all services

    .. tab:: Linux

        .. code-block::

            sudo python setup.py --mode uninstall --all --remove


    .. tab:: Mac

        .. code-block::

            sudo python setup.py --mode uninstall --all --remove

    .. tab:: Windows

        .. code-block::

            python setup.py --mode uninstall --all --remove



.. warning::
    Select services which you want to retain, all other services will be stopped.

Uninstall DLTK
===============

To uninstall OpenDLTK, run below commands.

.. tab:: selected services

    .. tab:: Linux

        .. code-block::

            sudo python setup.py --mode uninstall --partial --purge


    .. tab:: Mac

        .. code-block::

            sudo python setup.py --mode uninstall --partial --purge

    .. tab:: Windows

        .. code-block::

            python setup.py --mode uninstall --partial --purge



.. tab:: all services

    .. tab:: Linux

        .. code-block::

            sudo python setup.py --mode uninstall --all --purge


    .. tab:: Mac

        .. code-block::

            sudo python setup.py --mode uninstall --all --purge

    .. tab:: Windows

        .. code-block::

            python setup.py --mode uninstall --all --purge



