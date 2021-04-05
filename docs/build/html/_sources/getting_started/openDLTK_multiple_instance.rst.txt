******************************
OpenDLTK on multiple machines
******************************

.. contents:: Table of Contents
    :depth: 4
    :local:

About
=====


Getting Started
===============
For simple deployment of OpenDLTK on multiple instances we will use `Ansible <https://www.ansible.com/>`__.

**What is Ansible?**

Ansible is a simple automation tool that automates software application deployment, cloud provisioning, and configuration management.

It's a server orchestration tool that helps you to manage and control a large number of server nodes from single places called 'Control Machines'

So in this section we will use ansible to deploy openDLTK services on multiple machines as shown in below diagram.

.. image:: images/DLTK_ansible_diagram.jpg
    :align: center

.. todo::

    1. Replace above image
    2. Add content for configuring Databases



Pre-requisites
================
1. 5-8 ubuntu OS machines
2. Python3 installed on all the machines
3. root/admin privileges

Installation
=============


**1. Ansible Installation**

.. code-block::

    $ sudo apt update
    $ sudo apt install software-properties-common
    $ sudo add-apt-repository ppa:ansible/ansible-2.9
    $ sudo apt install ansible

For more detailed installation guide, please refer this `link <https://docs.ansible.com/ansible/2.7/installation_guide/intro_installation.html>`__

**2. Clone openDLTK github repository**

.. code-block::

    git clone https://github.com/dltk-ai/openDLTK


**3. What this repo contains**

.. todo::

    add directory tree

Ansible playbooks & roles, docker-compose files for openDLTK services & configurations files.

**4. Details about Ansible Files**

.. glossary::
    Ansible Playbook
        These are a set of instructions that you send to run on a single or group of server hosts.

        The Ansible Playbook contains some basic configuration, including hosts and user information of the provision servers, a task list that will be implemented to deploy openDLTK services.

        Ansible playbooks and roles are present in this repository, which can be used directly.


    Hosts
        It's an inventory file that contains pieces of information about managed servers by ansible. It allows you to create a group of servers that make you more easier to manage and scale the inventory file itself.

    Ansible role
        It is a set of tasks to configure a host to serve a certain purpose like configuring a service. Roles are defined using YAML files with a predefined directory structure. A role directory structure contains directories: defaults, vars, tasks, files, templates, meta, handlers.



**2. Initialize DLTK setup**

.. code-block::

    cd openDLTK
    pip install -r requirements.txt
    sudo python3 setup.py -m init

**3. Updating Configuration**

    Please update config.env file saved at :file:`/usr/dltk-ai/config.env`

    **Ansible Host Configurations**


To verify whether ansible host & roles are setup correctly, we will use following commands

.. code-block::

    ansible -m ping all



*3.a Configuring Storage*

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


*3.b Configure supported AI Engines Credentials*

    .. tab:: Azure

        .. code-block::

            AZURE_LANGUAGE_SUBSCRIPTION_KEY="USER_DEFINED"
            AZURE_BASE_URL="USER_DEFINED"



    .. tab:: IBM

        .. code-block::

            IBM_LANGUAGE_URL="USER_DEFINED"
            IBM_SUBSCRIPTION_KEY="USER_DEFINED"

*3.c Authentication*

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

        sudo python3 setup.py -m update_config


**5. Install Services**

        **Docker**

        .. code-block::

            sudo ansible-playbook ansible/playbooks/dltk-ai-docker.yml --extra-vars "folderpath=home/dltk"

        **Database**

            .. tab:: Already Existing Postgres

                1. Please update your existing postgres details in **config.env**, if not already done in configuration step.

            .. tab:: Setup Postgres

                .. code-block::

                    # please go to openDLTK directory
                    sudo ansible-playbook ansible/playbooks/dltk-ai-postgres.yml --extra-vars "folderpath=home/dltk"


            To setup InfluxDB and Redis

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-db.yml --extra-vars "folderpath=home/dltk"

        **Base Services**

            Base Service will setup Kong, Registry Service (Eureka), Solution Service.

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-base.yml --extra-vars "folderpath=home/dltk"

        .. warning::

            Database and Base are necessary to run below services, so proceed to other service deployment after deploying above two services.

        **Machine Learning**



            ML wrapper installation Steps

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-ml-wrapper.yml --extra-vars "folderpath=home/dltk"

            .. tab:: ML Scikit

                .. code-block::

                    sudo ansible-playbook ansible/playbooks/dltk-ai-ml-scikit.yml --extra-vars "folderpath=home/dltk"

            .. tab:: ML H2O

                .. code-block::

                    sudo ansible-playbook ansible/playbooks/dltk-ai-ml-h2o.yml --extra-vars "folderpath=home/dltk"

            .. tab:: ML Weka

                sudo ansible-playbook ansible/playbooks/dltk-ai-ml-weka.yml --extra-vars "folderpath=home/dltk"



        **Computer Vision**

            For running Computer vision services we will first deploy a wrapper which route the Images, client request to right processor

            To install Computer Vision Wrapper, run below command

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-cv-wrapper.yml --extra-vars "folderpath=home/dltk"


            .. tab:: Image Classification

                Image Classification takes Image as an input & return predicted labels as output in JSON format

                To run Image Classification service, run below command

                .. code-block::

                    sudo ansible-playbook ansible/playbooks/dltk-ai-cv-image-classification.yml --extra-vars "folderpath=home/dltk"

                .. seealso::
                    For more details on Image Classification features, please refer this section


            .. tab:: Object Detection

                Object Detection detects Objects in an Image

                To deploy Object Detection service, run below command in ansible control machine

                .. code-block::

                    sudo ansible-playbook ansible/playbooks/dltk-ai-cv-object-detection.yml --extra-vars "folderpath=home/dltk"

                .. seealso::
                        For more details on Object Detection features, please refer this section

            .. tab:: Face Analytics

                This service provide state-of-the-art open source AI models & support to various AI engines to provide face analytics on Images

                To deploy Face Analytics services, run below command in ansible control machine

                .. code-block::

                    sudo ansible-playbook ansible/playbooks/dltk-ai-cv-face-analytics.yml --extra-vars "folderpath=home/dltk"

                .. seealso::
                        For more details on Face Analytics features, please refer this section


        **Natural Language Processing**

            This service provide various NLP features like Name Entity Recognition, Part of Speech and Sentiment Analysis using various open source AI models & supported AI Engines

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-nlp.yml --extra-vars "folderpath=home/dltk"

            .. seealso::

                For more detail on NLP features, please refer this section

Usage
===============

.. note:: Below code block can be run after DLTK python client SDK is installed. The installation of which is covered in next section here `installation <pythonclientsdk.html#installation>`_ .

.. tab:: with Auth Disabled

    .. code-block::

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response.text)


.. tab:: with Auth Enabled

    .. code-block::

        import dltk_ai
        client = dltk_ai.DltkAiClient('86122578-4b01-418d-80cc-049e283d1e2b', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response.text)

.. seealso::
    1. To enable/disable authentication `link <http://localhost:63342/qubitai-dltk/docs/build/html/getting_started/toggle_auth.html>`__ .
    2. How to Create user and Generate API Key `link <http://localhost:63342/qubitai-dltk/docs/build/html/getting_started/generateAPIkey.html>`__ .


