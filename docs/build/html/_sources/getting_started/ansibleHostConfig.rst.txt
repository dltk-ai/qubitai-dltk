
Configure Ansible Hosts
-----------------------


The host file is used to store connections for Anisble playbooks. There are options to define connection parameters:

        ``ansible_host`` is the hostname or IP address

        ``ansible_port`` is the port the machine uses for SSH

        ``ansible_user`` is the remote user to connect as

        ``ansible_ssh_pass`` if using a password to SSH

        ``ansible_ssh_private_key_file`` if you need to use multiple keys that are specific to hosts

        These are the most commonly used options. More can be found in the `Ansible official documentation <https://docs.ansible.com/ansible/latest/user_guide/index.html>`__.

