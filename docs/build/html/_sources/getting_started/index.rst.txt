**********
About DLTK
**********

DLTK is an enterprise ready cloud platform that offers self-service AI capabilities. Qubit AI’s philosophy is to empower businesses with cutting edge technologies.

*******************
Creating an Account
*******************

1. Click on Sign Up and create a new account on https://cloud.dltk.ai/

2. You will be getting a verification email to the registered mail id. Please click on "Verify Now" button in the mail.

3. After verifying the email, you will be redirected to dltk login page. 

4. Enter your credentials to gain access


***************
Creating an App
***************

ApiKey Generation
-----------------

1. Once you login, to access DLTK Cloud, click on the Console button or the console tab available in the menu bar.

2. Click on the Projects menu dropdown and select Create App

3. Enter the name of the app and a short description

4. The dashboard will be visible with the given name and description along with an API key


***********
Setup Guide
***********

Installation
------------

Installing From Source
^^^^^^^^^^^^^^^^^^^^^^

* Clone the repo
``git clone https://github.com/dltk-ai/qubitai-dltk.git``

* Set working directory to qubitai-dltk folder

* Install requirements from requirements.txt file
``pip install -r requirements.txt``

Installing through pip
^^^^^^^^^^^^^^^^^^^^^^

``pip install qubitai-dltk``


Initializing the DLTK Client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setting up your ApiKey**::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')








