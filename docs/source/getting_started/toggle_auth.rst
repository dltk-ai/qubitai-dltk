Authentication
==============

Enable Authentication
----------------------


.. tab:: Single Instance

    .. tab:: Linux

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

    .. tab:: Mac

        Please update config.env file saved at ``/usr/dltk-ai/config.env``

    .. tab:: Windows

        Please update **config.env** file saved at ``C:\Users\{username}\AppData\Local\dltk_ai\config.env``

    .. code-block::

        AUTH_ENABLED="true"

        # SMTP setup
        SMTP_HOST="YOUR_SMTP_HOST"
        SMTP_PORT=587
        SMTP_USERNAME="YOUR_SMTP_USERNAME"
        SMTP_PASSWORD="YOUR_SMTP_USER_PASSWORD"

        # UI SERVER URL(replace localhost with server IP in case of remote machine)
        UI_SERVICE_URL="http://localhost:8082"

    .. code-block::

        python setup.py --mode auth --auth True

.. tab:: Multiple Instances

    Please update config.env file saved at ``/usr/dltk-ai/config.env``

    1. update ``AUTH_ENABLED="true"``
    2.



