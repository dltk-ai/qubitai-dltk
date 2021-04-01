DLTK Python SDK Client Installation
------------------------------------

.. contents:: Table of Contents
    :depth: 4
    :local:


.. _python_sdk_installation:

Installation
============

.. tab:: Installing through pip

    The simplest way to install python client SDK is using below pip command

    .. code-block:: bash

            pip install qubitai-dltk

    .. note::

        The version needs to be matched with the version of openDLTK docker container being used.

        So for example if you installed **openDLTK version 1.2** then you can install latest version of **qubitai-dltk==1.2.x**

.. tab:: Installing From Source

    .. code-block:: shell

        # Clone the repo
        git clone https://github.com/dltk-ai/qubitai-dltk.git

        # Change working directory
        cd qubitai-dltk

        # Install requirements from requirements.txt file
        pip install -r requirements.txt



Initializing the DLTK Client
==============================

Now you can use your favourite IDEs (Jupyter Notebook, pycharm, VS code) to initialize DLTK client using below commands.


.. code-block:: python

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://12.1.1.8:8000')

