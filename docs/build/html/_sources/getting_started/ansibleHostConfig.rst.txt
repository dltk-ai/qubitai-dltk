
Configure Ansible Hosts
-----------------------


The host file is used to store connections for Anisble playbooks. There are options to define connection parameters:

        ``ansible_host`` is the hostname or IP address

        ``ansible_port`` is the port the machine uses for SSH

        ``ansible_user`` is the remote user to connect as

        ``ansible_ssh_pass`` if using a password to SSH

        ``ansible_ssh_private_key_file`` if you need to use multiple keys that are specific to hosts

        These are the most commonly used options. More can be found in the Ansible official documentation.

.. tab:: Various Host File Configurations

    1. **Without SSH key**

    2. **With SSH key and without root password**

    3. **With SSH key and with root password**


.. warning::

    1. Please modify only ``12.1.1.9 ansible_user=dltk ansible_become=yes ansible_become_password=<ROOT_USER_PASSWORD> ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa``
    2. Please don't modify Host names (e.g ``dltk-ai-db-host``, ``dltk-ai-object-detector-host``)


.. seealso::

    Add link to Ansible

