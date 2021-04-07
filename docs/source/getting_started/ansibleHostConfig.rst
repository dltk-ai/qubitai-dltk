
Configure Ansible Hosts
-----------------------


The host file is used to store connections for Anisble playbooks. There are options to define connection parameters:

        ``ansible_host`` is the hostname or IP address

        ``ansible_port`` is the port the machine uses for SSH

        ``ansible_user`` is the remote user to connect as

        ``ansible_ssh_pass`` if using a password to SSH

        ``ansible_ssh_private_key_file`` if you need to use multiple keys that are specific to hosts

        These are the most commonly used options. More can be found in the `Ansible official documentation <https://docs.ansible.com/ansible/latest/user_guide/index.html>`__.

|

Ansible Host File Configuration for Machines with Private Key
==============================================================

    Replace ``XX.XX.XX`` with IP address of host machine and ``USER_PASSWORD`` with your machine's password and update ``/path/to/private/key/file``

    .. code-block::

        [dltk-ai-db-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

        [dltk-ai-base-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

        [dltk-ai-wrapper-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

        [dltk-ai-ml-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

        [dltk-ai-image-processor-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file

        [dltk-ai-object-detector-host]
        XX.XX.XX.XX ansible_user=dltk ansible_become=yes ansible_become_password=USER_PASSWORD ansible_ssh_private_key_file=/path/to/private/key/file



.. seealso::

    Ansible Official Documentation - `Connecting to hosts: behavioral inventory parameters <https://docs.ansible.com/ansible/latest/user_guide/intro_inventory.html#connecting-to-hosts-behavioral-inventory-parameters>`__

