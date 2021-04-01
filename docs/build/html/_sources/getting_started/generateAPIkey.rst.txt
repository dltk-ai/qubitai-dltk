********************************
Create user and Generate APIkey
********************************

.. contents:: Table of Contents
    :depth: 4
    :local:


The UI for user creation can be accesed at `http://localhost:8082 <http://localhost:8082>`__ ,by clicking on `Sign Up` button on top right, user will be able to view below pages


SIGN UP
=============================
 `Sign Up Page <http://localhost:8082/sign_up/>`__

.. image:: images/signup.png

After filling above details, click on `Sign Up` button.
User will receive an Email for verification, on successful verification of email, user will be redirected to `Sign In` page.

EMAIL VERIFICATION
=============================

.. image:: images/verification_mail.png

SIGN IN
=============================
 `Sign In Page <http://localhost:8082/sign_in/>`__

.. image:: images/login.png

CREATE AN PROJECT
=============================

After successful login, user will be able to view project list page

.. image:: images/project_list_page.png

Next user needs to create a project, by giving details as shown in below image

.. image:: images/create_project.png


GET AN API KEY
=============================

After successful creation of project, you can go to Project Details Page to get `API Key`

.. image:: images/project_details_page.png

USE SDK WITH API KEY
=============================

Using above generated API key you can use in python SDK client

.. code-block::

        import dltk_ai

        client = dltk_ai.DltkAiClient('86122578-4b01-418d-80cc-049e283d1e2b', base_url='http://localhost:8000')

        text = "The product is very easy to use and has got a really good life expectancy."

        sentiment_analysis_response = client.sentiment_analysis(text)

        print(sentiment_analysis_response.text)

Expected Output:

.. code-block::

        {
          "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
        }


for more detailed information on `Qubitai-DLTK python SDK <https://github.com/dltk-ai/qubitai-dltk>`__ features & usage please refer to this `Documentation <https://docs.dltk.ai>`__