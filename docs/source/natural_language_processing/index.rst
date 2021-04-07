*****
About
*****

DLTK's NLP service has pre-built Models for various Natural Language Processing tasks, which can leverage to predict on input text. Third-party APIs like IBM and Azure, too, are integrated.

******************
Sentiment Analysis
******************

Detects positive or negative sentiment in text.

Usage: social media monitoring, Brand reputation analysis, etc.


.. py:function:: client.sentiment_analysis(text, sources = ['spacy'])

   :param text: text for analysing the sentiment
   :param sources: algorithm to use - azure/ibm_watson/spacy/model_1. Default is spacy.
   :rtype: A json object containing the sentiment analysis response

**Example**

.. code-block:: python

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "The product is very easy to use and has got a really good life expectancy."

    sentiment_analysis_response = client.sentiment_analysis(text)

    print(sentiment_analysis_response.text)

**Output**

.. code-block:: JSON

    {
      "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
    }



**********************
Parts of Speech Tagger
**********************

It identifies and marks a word in a text as corresponding to a particular part of speech, based on its definition and context.

.. py:function:: client.pos_tagger(text, sources = ['spacy'])

   :param text: text for Parts of Speech Tagging
   :param sources: algorithm to use - ibm_watson/spacy
   :rtype: A json object containing the Parts of Speech of the words in the sentence.

**Example**

.. code-block:: python

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "They refuse to permit us to obtain the refuse permit."

    pos_tagger_response = client.pos_tagger(text)

    print(pos_tagger_response)


**Output**

.. code-block:: JSON

    {
      "spacy": {"result": {"They": "PRP", "refuse": "NN", "to": "TO", "permit": "NN", "us": "PRP", "obtain": "VB", "the": "DT", ".": "."}}
    }


************************
Named Entity Recognition
************************

It identifies key information (entities) in text. Each token is given an appropriate tag such as Person, Location, Organisation, etc.

.. py:function:: client.ner_tagger(text, sources = ['spacy'])

   :param text: text for Named Entity Recognition
   :param sources: algorithm to use - azure/ibm_watson/spacy
   :rtype: A json object with Entities identified in the given text.

**Example**

    .. code-block:: python

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY')

        text = "John has moved to California recently."

        ner_tagger_response = client.ner_tagger(text)

        print(ner_tagger_response)

**Output**

.. code-block:: JSON

    {
      "spacy": {"result": {"John": "PERSON", "California": "GPE"}, "persons": [], "organizations": []}
    }


*****************
Dependancy Parser
*****************

It analyses the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads.

Usage: Grammar monitoring.

.. py:function:: client.dependency_parser(text):

   :param text: text for dependency parser
   :rtype: A json object with Entities identified in the given text.

**Example**

    .. code-block:: python

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY')

        text = "And now for something completely different."

        dependency_parser_response = client.dependency_parser(text)

        print(dependency_parser_response)

**Output**

    .. code-block:: JSON

        {
         'And': {'dep': 'cc', 'headText': 'for', 'headPOS': 'ADP', 'children': []},
         'now': {'dep': 'advmod', 'headText': 'for', 'headPOS': 'ADP', 'children': []},
         'for': {'dep': 'ROOT','headText': 'for', 'headPOS': 'ADP', 'children': ['And', 'now', 'something', '.']},
         'something': {'dep': 'pobj', 'headText': 'for', 'headPOS': 'ADP', 'children': ['different']},
         'completely': {'dep': 'advmod', 'headText': 'different', 'headPOS': 'ADJ', 'children': []},
         'different': {'dep': 'amod','headText': 'something', 'headPOS': 'NOUN', 'children': ['completely']},
         '.': {'dep': 'punct', 'headText': 'for', 'headPOS': 'ADP', 'children': []}
         }


****************
Tags Recognition
****************

It identifies the important words in a sentence.

.. py:function:: client.tags(text)

   :param text: text for tags recognition
   :rtype: A json object with Tags identified in the given text.

**Example**

    .. code-block:: python
        :linenos:

        import dltk_ai
        client = dltk_ai.DltkAiClient('YOUR_API_KEY')

        text = "Elon Musk has shared a photo of the spacesuit designed by SpaceX. This is the second image shared of the new design and the first to feature the spacesuit full-body look.."

        tags_response = client.tags(text)

        print(tags_response)

**Output**

    .. code-block:: JSON

        {
          "rake": {"tags": ["elon musk", "shared", "photo", "spacesuit designed", "spacex", "image shared", "design", "feature", "spacesuit full", "body"]}
        }

