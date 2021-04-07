Authentication
==============

If you have group of developers using OpenDLTK, its desirable to enable Authentication. So each developer can register themselves
to the OpenDLTK Portal by providing their Basic Details like Username & EmailID.

Developers can Sign Up & Create Project using ``User Management UI``. as shown in `User Registration and API key generation page <generateAPIkey.html>`__


.. contents:: Table of Contents
    :depth: 4
    :local:

Single Machine
---------------

-----------

Enable Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~


    To Enable Authentication

    .. tab:: Linux

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

        .. code-block::

            AUTH_ENABLED="true"

            # SMTP setup
            SMTP_HOST="YOUR_SMTP_HOST"
            SMTP_PORT=587
            SMTP_USERNAME="YOUR_SMTP_USERNAME"
            SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

            # UI SERVER URL(replace localhost with server IP in case of remote machine)
            UI_SERVICE_URL="http://localhost:8082"

        .. code-block:: console

            $ sudo python3 setup.py -m update_config
            $ sudo python3 setup.py --mode auth --auth True

    .. tab:: Mac

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

        .. code-block::

            AUTH_ENABLED="true"

            # SMTP setup
            SMTP_HOST="YOUR_SMTP_HOST"
            SMTP_PORT=587
            SMTP_USERNAME="YOUR_SMTP_USERNAME"
            SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

            # UI SERVER URL(replace localhost with server IP in case of remote machine)
            UI_SERVICE_URL="http://localhost:8082"

        .. code-block:: console

            $ sudo python3 setup.py -m update_config
            $ sudo python3 setup.py --mode auth --auth True

    .. tab:: Windows

        Please update **config.env** file saved at ``C:\Users\{username}\AppData\Local\dltk-ai\config.env``

        .. code-block::

            AUTH_ENABLED="true"

            # SMTP setup
            SMTP_HOST="YOUR_SMTP_HOST"
            SMTP_PORT=587
            SMTP_USERNAME="YOUR_SMTP_USERNAME"
            SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

            # UI SERVER URL(replace localhost with server IP in case of remote machine)
            UI_SERVICE_URL="http://localhost:8082"

        .. code-block:: console

            $ python setup.py -m update_config
            $ python setup.py --mode auth --auth True


Disable Authentication
~~~~~~~~~~~~~~~~~~~~~~


    To Disable Authentication,

    .. tab:: Linux

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

        .. code-block::

            AUTH_ENABLED="false"


    .. tab:: Mac

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

        .. code-block::

            AUTH_ENABLED="false"


    .. tab:: Windows

        Please update **config.env** file saved at ``C:\Users\{username}\AppData\Local\dltk-ai\config.env``

        .. code-block::

            AUTH_ENABLED="false"




Multiple Machines
------------------

------------

Enabling Authentication
~~~~~~~~~~~~~~~~~~~~~~~

        Please update config.env file saved at ``/usr/dltk-ai/config.env`` with following details

        .. code-block::

            AUTH_ENABLED="true"

            # SMTP setup
            SMTP_HOST="YOUR_SMTP_HOST"
            SMTP_PORT=587
            SMTP_USERNAME="YOUR_SMTP_USERNAME"
            SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

            # UI SERVER URL(replace localhost with server IP in case of remote machine)
            UI_SERVICE_URL="http://dltk-ai-base-host-ipaddress:8082"


        To update configurations changes, run below command

        .. code-block:: console

            $ sudo python3 setup.py -m update_config

        Run below command to restart the base services with updated configurations which will ensure that only request with valid API key will served.

        .. code-block:: console

            $ sudo ansible-playbook ansible/playbooks/dltk-ai-enable-auth.yml --extra-vars "folderpath=/path/to/dltk"



Disabling Authentication
~~~~~~~~~~~~~~~~~~~~~~~~

        Please update config_multi.env file saved at /usr/dltk-ai/config_multi.env with following details

        .. code-block::

            AUTH_ENABLED="false"


        To update configurations changes, run below command

        .. code-block:: console

            $ sudo python3 setup.py -m update_config

        Run below command to restart the base services with updated configurations which will disable need of providing a valid API key, so request by any developer will be served by OpenDLTK.

        .. code-block:: console

            $ sudo ansible-playbook ansible/playbooks/dltk-ai-enable-auth.yml --extra-vars "folderpath=/path/to/dltk"
