.. _openDLTK-multiple-machine-setup:
******************************
OpenDLTK on multiple machines
******************************

.. contents:: Table of Contents
    :depth: 4
    :local:

About
=====

As these AI models are computationally heavy and require higher compute power, its difficult to run all the services to run on local or single machine,
So in order to run these services smoothly its recommended to deploy OpenDLTK on multiple machines.

.. seealso::
    - `OpenDLTK Services <index.html#opendltk-and-python-sdk-client>`__ for more details on services present in OpenDLTK



Getting Started
===============

For simple deployment of OpenDLTK on multiple instances we will use `Ansible <https://www.ansible.com/>`__.


.. image:: images/ansible_intro.png
    :align: right
    :width: 400


**What is Ansible?**

Ansible is a simple automation tool that automates software application deployment, cloud provisioning, and configuration management.

It's a server orchestration tool that helps you to manage and control a large number of `remote servers`  from single machine called `Control Machines` where ansible is installed.



    **1. Ansible Playbook**
        These are a set of instructions that you send to run on a single or group of server hosts.

    **2. Hosts**
        It's an inventory file that contains pieces of information about managed servers by ansible.

    **3. Ansible role**
        It is a set of tasks to configure a host to serve a certain purpose like configuring a service.

.. seealso::

    `How Ansible Works? <https://www.ansible.com/overview/how-ansible-works>`__


We will use ansible to deploy OpenDLTK services on multiple machines as shown in below diagram.


.. image:: images/ansible_install_diagram.png
    :align: center
    :width: 1000



Pre-requisites
================
- 5 to 8 Ubuntu Machines with 2 vCPUs, 8 GB memory, 30GB Disk Space configurations for each machine
- Configure VPC and all the remote machine should be in private subnet, with only private IP enabled for every remote machine.
- Python3 installed on all the machines
- Root/Admin privileges

Installation
=============


**1. Ansible Installation**

.. code-block:: shell-session

    $ sudo apt update
    $ sudo apt install software-properties-common
    $ sudo add-apt-repository ppa:ansible/ansible-2.9
    $ sudo apt install ansible

To verify whether ansible installation is successful, run below command

.. code-block:: shell-session

    $ sudo ansible --version

*Expected Output*

.. code-block:: shell-session

    ansible 2.9.6
    config file = /etc/ansible/ansible.cfg
    configured module search path = ['/root/.ansible/plugins/modules', '/usr/share/ansible/plugins/modules']
    ansible python module location = /usr/lib/python3/dist-packages/ansible
    executable location = /usr/bin/ansible
    python version = 3.8.5 (default, Jul 28 2020, 12:59:40) [GCC 9.3.0]

For more detailed installation guide, please refer this `link <https://docs.ansible.com/ansible/2.7/installation_guide/intro_installation.html>`__

**2. Clone openDLTK github repository**

.. code-block:: console

    $ git clone https://github.com/dltk-ai/openDLTK


**3. What this repo contains**

This repo contains ``Ansible playbooks & roles``, ``docker-compose`` files for OpenDLTK microservices & necessary ``configurations`` files.

.. code-block::

    ├── ansible
    │    └── playbooks
    |    |    ├── dltk-ai-base.yml
    |    |    ├── dltk-ai-cv-face-analytics.yml
    |    |    ├── dltk-ai-cv-image-classification.yml
    |    |    ├── dltk-ai-cv-object-detection.yml
    |    |    ├── dltk-ai-cv-wrapper.yml
    |    |    ├── dltk-ai-db.yml
    |    |    ├── dltk-ai-disable-auth-1.yml
    |    |    ├── dltk-ai-disable-auth-2.yml
    |    |    ├── dltk-ai-disable-auth-db-migrations.yml
    |    |    ├── dltk-ai-disable-auth.yml
    |    |    ├── dltk-ai-disable-web.yml
    |    |    ├── dltk-ai-docker.yml
    |    |    ├── dltk-ai-enable-auth-1.yml
    |    |    ├── dltk-ai-enable-auth-2.yml
    |    |    ├── dltk-ai-enable-auth-db-migrations.yml
    |    |    ├── dltk-ai-enable-auth.yml
    |    |    ├── dltk-ai-ml-h2o.yml
    |    |    ├── dltk-ai-ml-scikit.yml
    |    |    ├── dltk-ai-ml-weka.yml
    |    |    ├── dltk-ai-ml-wrapper.yml
    |    |    ├── dltk-ai-nlp.yml
    |    |    ├── dltk-ai-postgres.yml
    |    |    ├── dltk-ai-stop-all.yml
    |    |    ├── dltk-ai-stop-base.yml
    |    |    ├── dltk-ai-stop-cv-face-analytics.yml
    |    |    ├── dltk-ai-stop-cv-image-classification.yml
    |    |    ├── dltk-ai-stop-cv-object-detection.yml
    |    |    ├── dltk-ai-stop-cv-wrapper.yml
    |    |    ├── dltk-ai-stop-ml-h2o.yml
    |    |    ├── dltk-ai-stop-ml-scikit.yml
    |    |    ├── dltk-ai-stop-ml-weka.yml
    |    |    ├── dltk-ai-stop-ml-wrapper.yml
    |    |    ├── dltk-ai-stop-nlp.yml
    |    |    ├── dltk-ai-stop-web.yml
    |    |    ├── dltk-ai-web.yml
    |    |    └── roles
    |    ├── base
    |    │   ├── registry-config
    |    │   └── solution-config
    |    ├── cv
    |    │   ├── face_analytics
    |    │   ├── pretrained_detectors
    |    │   └── wrapper
    |    ├── db
    |    ├── docs
    |    ├── ml
    |    ├── nlp
    |    ├── pgdump
    |    ├── utils
    |    └── web



**2. Initialize DLTK setup**

    .. code-block:: console

        $ cd openDLTK

    Use the following command to install pip for Python3 and install the necessary packages required for installing OpenDLTK installer.

    .. code-block:: console

        $ sudo apt install python3-pip
        $ sudo pip3 install -r requirements.txt

    Run below command to initialize OpenDLTK installation, this will create a config file at :file:`/usr/dltk-ai/config_multi.env` which is required to manage all the configurations required for installation.

    .. code-block::

        $ sudo python3 setup_init.py -m init

**3. Updating Configuration**

    **Update config_multi.env**

        Please update ``config_multi.env`` file saved at :file:`/usr/dltk-ai/config_multi.env` by referring to `Configurations Details <configurations.html>`__

**4. Ansible Host Configurations**

    While installing Ansible a hosts file is generated at ``/etc/ansible/`` path

    Copy below host file into ``/etc/ansible/hosts`` path

    Update following details in below file
        - ``XX.XX.XX.XX`` with IP address of host machine
        - ``USER_PASSWORD`` with your machine's password
        - ``/path/to/private/key/file`` with path to your private key
        - ``root_username`` with your username having admin privileges


        .. code-block::

            [dltk-ai-db-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

            [dltk-ai-base-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

            [dltk-ai-wrapper-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

            [dltk-ai-ml-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

            [dltk-ai-image-processor-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

            [dltk-ai-object-detector-host]
            XX.XX.XX.XX ansible_user=root_username ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file


        .. caution::

            Please don't modify host names like (``dltk-ai-object-detector-host``, ``dltk-ai-db-host``)

        .. note::

            For more detail on configuring Remote Machines for Ansible, please refer to `Ansible Connection Setup Guide <ansibleHostConfig.html>`__


        Generate a SSH key and copy to remote machine
            a. Generate an SSH Key
                With OpenSSH, an SSH key is created by running ``ssh-keygen`` command which generates public/private rsa key pair.

            b. Copy the key to a server
                Once an SSH key has been created, the ssh-copy-id command can be used to install it as an authorized key on the remote machine. Once the key has been authorized for SSH, it grants access to the remote machine without a password.

                Use command ``ssh-copy-id -i ~/.ssh/mykey user@host`` to copy SSH key.

                This logs into the remote machine, and copies keys to the remote machine, and configures them to grant access by adding them to the authorized_keys file. The copying may ask for a password or other authentication for the server.
                Only the public key is copied to the remote machine.

        Please login to **all** remote machines using ``ssh username@IPaddress`` command from ansible machine

        To verify whether ansible host & roles are setup correctly, we will use following commands


        .. code-block:: console

            $ ansible -m ping all

        Expected Output

        .. code-block::

            XX.XX.XX.XX | SUCCESS => {
            "ansible_facts": {
                "discovered_interpreter_python": "/usr/bin/python3"
            },
            "changed": false,
            "ping": "pong"
            }
            :



**4. Update config**

    .. code-block:: console

        $ sudo python3 setup_init.py -m update_config

    .. important::

        Whenever config_multi.env is changed this command needs to be run, to update those changes.


**5. Install Services**

        Please provide ``folderpath`` where you want to install OpenDLTK services on remote machines in all the below commands.

        .. tip::

            Please use same path in all the remote machines

        **Docker**

            To install docker on all the remote machine, below ansible playbook command can be used. This will install docker on all the remote machines.

            .. code-block:: console

                $ sudo ansible-playbook ansible/playbooks/dltk-ai-docker.yml --extra-vars "folderpath=/path/to/folder"


        **Database**


            **Postgres Setup**

                If you already have a postgres database then you can skip setting up a new postgres container, the details of existing postgres
                needs to be updated in :file:`/usr/dltk-ai/config_multi.env` file.

                But in case you dont have an existing postgres database, you need to setup postgres database.

                .. tab:: Already Existing Postgres

                    1. Please update your existing postgres details in :file:`/usr/dltk-ai/config_multi.env`, if not already done in configuration step.

                    2. After Updating :file:`/usr/dltk-ai/config_multi.env` , run ``sudo python3 setup.py -m update_config`` command to update configurations changes.

                .. tab:: Setup Postgres

                    Run below command to setup postgres container

                    .. code-block:: console

                        # please go to openDLTK directory
                        $ sudo ansible-playbook ansible/playbooks/dltk-ai-postgres.yml --extra-vars "folderpath=/path/to/folder"


            **InfluxDB and Redis Setup**

                To setup Influxdb and Redis containers on remote machines, run below command.

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-db.yml --extra-vars "folderpath=/path/to/folder"

        **Base Services**

            To setup Base Service containers on remote machines, run below command.

            Base Service will setup Kong, Registry Service, Solution Service.

            .. code-block:: console

                $ sudo ansible-playbook ansible/playbooks/dltk-ai-base.yml --extra-vars "folderpath=/path/to/folder"

        .. warning::

            Database and Base are necessary to run below services, so proceed to other service deployment after deploying above two services.

        **Machine Learning**


            To setup ML Wrapper Service container on remote machine, run below command

            .. code-block:: console

                $ sudo ansible-playbook ansible/playbooks/dltk-ai-ml-wrapper.yml --extra-vars "folderpath=/path/to/folder"

            Based on your choice to install ML-Scikit, ML-H2O or ML-weka, run below command respectively.


            .. tab:: ML Scikit

                To setup ML Scikit Service container on remote machine, run below command

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-ml-scikit.yml --extra-vars "folderpath=/path/to/folder"

            .. tab:: ML H2O

                To setup ML H2O Service container on remote machine, run below command

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-ml-h2o.yml --extra-vars "folderpath=/path/to/folder"

            .. tab:: ML Weka

                To setup ML Weka Service container on remote machine, run below command

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-ml-weka.yml --extra-vars "folderpath=/path/to/folder"



        **Computer Vision**

            For running Computer vision services we will first deploy a wrapper which route the Images, client request to right processor

            To setup Computer Vision Wrapper Service container on remote machine, run below command

            .. code-block:: console

                $ sudo ansible-playbook ansible/playbooks/dltk-ai-cv-wrapper.yml --extra-vars "folderpath=/path/to/folder"

            Based on your choice to install Image Classification, Object Detection, Face Analytics run below command respectively.

            .. tab:: Image Classification

                Image Classification takes Image as an input & return predicted labels as output in JSON format

                To setup Computer Vision Image Classification Service container on remote machine, run below command

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-cv-image-classification.yml --extra-vars "folderpath=/path/to/folder"

                .. seealso::
                    For more details on Image Classification features, please refer this section


            .. tab:: Object Detection

                Object Detection detects Objects in an Image

                To deploy Object Detection service, run below command in ansible control machine

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-cv-object-detection.yml --extra-vars "folderpath=/path/to/folder"

                .. seealso::
                        For more details on Object Detection features, please refer this section

            .. tab:: Face Analytics

                This service provide state-of-the-art open source AI models & support to various AI engines to provide face analytics on Images

                To deploy Face Analytics services, run below command in ansible control machine

                .. code-block:: console

                    $ sudo ansible-playbook ansible/playbooks/dltk-ai-cv-face-analytics.yml --extra-vars "folderpath=/path/to/folder"

                .. seealso::
                        For more details on Face Analytics features, please refer this section


        **Natural Language Processing**

            This service provide various NLP features like Name Entity Recognition, Part of Speech and Sentiment Analysis using various open source AI models & supported AI Engines

            .. code-block:: console

                $ sudo ansible-playbook ansible/playbooks/dltk-ai-nlp.yml --extra-vars "folderpath=/path/to/folder"

            .. seealso::

                For more detail on NLP features, please refer this section

**OpenDLTK Services Status Check**

    All the OpenDLTK Service will register to registry service while starting and also every 30sec update their status.
    To check whether services installed are correctly started or not, go to http://your_base_ip_address:8761 and check whether your services are registered or not.

    *Expected Output*

    .. image:: images/eureka.png
        :align: center


    As we can see in this example, Machine Learning Service & Machine Learning Weka Service & Solution Service are setup correctly.


Usage
===============

.. note::

    Below code block can be run after DLTK python client SDK is installed. The installation of which is covered in next section here `installation <pythonclientsdk.html#installation>`_ .
    Its recommended to map a DNS name to the kong IP address i.e YOUR_BASE_IP_ADDRESS:8000


.. tab:: with Auth Disabled

    .. code-block::

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://your-dltk-ai-base-host:8000')

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
    1. To enable/disable authentication `link <toggle_auth.html>`__ .
    2. How to Create user and Generate API Key `link <generateAPIkey.html>`__ .


Stop DLTK Services
======================

    **Natural Language Processing**

        .. code-block::

             sudo ansible-playbook ansible/playbooks/dltk-ai-stop-nlp.yml --extra-vars "folderpath=/path/to/folder"

    **Machine Learning**

        .. tab:: ML scikit

            .. code-block::

                 sudo ansible-playbook ansible/playbooks/dltk-ai-stop-ml-scikit.yml --extra-vars "folderpath=/path/to/folder"

        .. tab:: ML H2O

            .. code-block::

                 sudo ansible-playbook ansible/playbooks/dltk-ai-stop-ml-h2o.yml --extra-vars "folderpath=/path/to/folder"


        .. tab:: ML Weka

            .. code-block::

                 sudo ansible-playbook ansible/playbooks/dltk-ai-stop-ml-weka.yml --extra-vars "folderpath=/path/to/folder"

        *ML Wrapper*

        .. caution::

            Run Below command to stop **ML-Wrapper** only if all the above ML services (ML Scikit, ML H2O, ML weka) are stopped.

        .. code-block::

                 sudo ansible-playbook ansible/playbooks/dltk-ai-stop-ml-wrapper.yml --extra-vars "folderpath=/path/to/folder"

    **Computer Vision**

        .. tab:: Image Classification

            To stop Image Classification service, run below command

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-stop-cv-image-classification.yml --extra-vars "folderpath=/path/to/folder"



        .. tab:: Object Detection


            To stop Object Detection service, run below command in ansible control machine

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-stop-cv-object-detection.yml --extra-vars "folderpath=/path/to/folder"


        .. tab:: Face Analytics


            To stop Face Analytics services, run below command in ansible control machine

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-stop-cv-face-analytics.yml --extra-vars "folderpath=/path/to/folder"

        *CV-Wrapper*

            To stop CV wrapper, run below command in ansible control machine

            .. caution::

                Run below command only if all the above computer vision services like Image Classification, Object Detection & Face Analytics are stopped.

            .. code-block::

                sudo ansible-playbook ansible/playbooks/dltk-ai-stop-cv-wrapper.yml --extra-vars "folderpath=/path/to/folder"

    **Base**

        .. caution::

                Run below command only to stop base service only if all the above services are stopped, as uninstalling base will impact all the DLTK services

        .. code-block::

            sudo ansible-playbook ansible/playbooks/dltk-ai-base.yml --extra-vars "folderpath=/path/to/folder"

    **Stop All**

        To stop all the OpenDLTK services, run below command

         .. code-block::

            sudo ansible-playbook ansible/playbooks/dltk-ai-stop-all.yml --extra-vars "folderpath=/path/to/folder"
