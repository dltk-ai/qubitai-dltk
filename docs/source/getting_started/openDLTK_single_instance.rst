*****************************
openDLTK on single instance
*****************************

.. contents:: Table of Contents
    :depth: 4
    :local:

Pre-requisites
================


.. tab:: Linux

    **1. Python**

    **2. Virtual Environment**

    **3. Docker**

    Please refer to `Linux docker installation guide <https://docs.docker.com/engine/install/>`__ for more detail on installation


.. tab:: Mac

    **1. Python**

    **2. Virtual Environment**

    **3. Docker**

    Please refer to `Mac docker installation guide <https://docs.docker.com/docker-for-mac/install/>`__

.. tab:: Windows

    **1. Python**

    **2. Virtual Environment**

    **3. Docker**

    Please refer to `Windows Home docker installation guide <https://docs.docker.com/docker-for-windows/install-windows-home/>`__

Installation
=============

**1. Clone openDLTK github repository**

.. code-block::

    git clone https://github.com/dltk-ai/openDLTK
    cd openDLTK
    pip install -r requirements.txt

**2. Initialize DLTK setup**

.. code-block::

    sudo python setup.py -m init

**3. Updating Configuration**

.. tab:: Linux

    Please update config.env file saved at ``/usr/dltk-ai/config.env``

.. tab:: Mac

    Please update config.env file saved at ``/usr/dltk-ai/config.env``

.. tab:: Windows

    Please update **config.env** file saved at ``C:\Users\{username}\AppData\Local\dltk_ai\config.env``


**3.a Configuring Storage**

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

.. tab:: Google Cloud Storage

    .. code-block::

        STORAGE_TYPE="gcp"

        # Values only for reference, replace with your details

        GCP_SERVICE_ACCOUNT_FILE=dltk-ai.json
        GCP_PRIVATE_BUCKET="dltk-ai-private"
        GCP_PUBLIC_BUCKET="dltk-ai-public"

.. tab:: Digital Ocean

    .. code-block::

        STORAGE_TYPE="do"

        # Values only for reference, replace with your credentials


        DO_ENDPOINT="sgp1.digitaloceanspaces.com"
        DO_ACCESS_KEY="SPZ4OSDVXC35R26"
        DO_SECRET_KEY="9b7SQmnFNx0vzAHWc5czKW75By01CH4"
        DO_BUCKET="dltk-ai"
        DO_REGION="sgp1"

.. warning::
    In case you decide to switch your initial storage from one source to another, the data migrations has to be handled by you.


**3.b Configure supported AI Engines Credentials**

.. tab:: Azure

    .. code-block::

        AZURE_LANGUAGE_SUBSCRIPTION_KEY="USER_DEFINED"
        AZURE_BASE_URL="USER_DEFINED"



.. tab:: IBM

    .. code-block::

        IBM_LANGUAGE_URL="USER_DEFINED"
        IBM_SUBSCRIPTION_KEY="USER_DEFINED"

**3.c Authentication**

.. tab:: Enable Authentication

    In config.env file, update

    .. code-block::

        AUTH_ENABLED="true"

    .. todo::
        If later you want to disable authentication, please refer this section

.. tab:: Disable Authentication

    In config.env file, update

    .. code-block::

        AUTH_ENABLED="false"

    .. todo::
        If later you want to enable authentication, please refer this section

**4. Update config**

    .. code-block::

        python setup.py -m update_config


**5. Install Services**

    .. code-block::

        python setup.py -m install


Usage
===============

.. note:: Below code block can be run after DLTK python client SDK is installed. The installation of which is covered in next section here `installation <pythonclientsdk.html#installation>`_ .

.. tab:: with Auth Enabled

    .. code-block::

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response.text)


.. tab:: with Auth Disabled

    .. code-block::

        import dltk_ai
        client = dltk_ai.DltkAiClient('86122578-4b01-418d-80cc-049e283d1e2b', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response.text)

.. todo::
    Update link of how to create a user & generate API key

Stop Services
===============

.. tab:: selected services

    .. code-block::

        python setup.py --mode uninstall --partial --remove


.. tab:: all services

    .. code-block::

        python setup.py --mode uninstall --all --remove

.. warning::
    Select services which you want to retain, all other services will be stopped.

Uninstall DLTK
===============

.. tab:: selected services

    .. code-block::

        python setup.py --mode uninstall --partial --purge

.. tab:: all services

    .. code-block::

        python setup.py --mode uninstall --all --purge

