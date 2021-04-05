
Configure Ansible Hosts
-----------------------



.. tab:: Host File

    .. code-block::

        [dltk-ai-postgres-host]
        12.1.1.8 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-db-host]
        12.1.1.8 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-base-host]
        12.1.1.9 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-wrapper-host]
        12.1.1.10 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-ml-wrapper-host]
        12.1.1.11 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-ml-scikit-host]
        12.1.1.11 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-ml-h2o-host]
        12.1.1.11 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-ml-weka-host]
        12.1.1.11 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-image-processor-host]
        12.1.1.12 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa

        [dltk-ai-object-detector-host]
        12.1.1.13 ansible_user=dltk ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa


.. tab:: Various Host File Configurations

    1. **Without SSH key**

    2. **With SSH key and without root password**

    3. **With SSH key and with root password**


.. warning::

    1. Please modify only ``12.1.1.9 ansible_user=dltk ansible_become=yes ansible_become_password=<ROOT_USER_PASSWORD> ansible_ssh_private_key_file=/home/dltk/.ssh/id_rsa``
    2. Please don't modify Host names (e.g ``dltk-ai-db-host``, ``dltk-ai-object-detector-host``)


.. seealso::

    Add link to Ansible

