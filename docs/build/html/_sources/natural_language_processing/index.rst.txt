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
    client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

    text = "I really like the new design of your website"

    dltk_sentiment = client.sentiment_analysis(text)

    print(dltk_sentiment)

**Output**

.. code-block:: JSON

    {
      "nltk_vader": {"emotion": "POSITIVE", "scores": {"compound": 0.4201, "negative": 0.0, "positive": 0.285, "neutral": 0.715}}
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
    client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

    text = "The old tired man was sitting under a tree and patiently waiting for his son to arrive"

    dltk_pos_tagger = client.pos_tagger(text)

    print(dltk_pos_tagger)


**Output**

.. code-block:: JSON

    {
      "spacy": {
          "result": {
              "The": "DT", "old": "JJ", "tired": "JJ", "man": "NN", "was": "VBD", "sitting": "VBG", "under": "IN", "a": "DT", "tree": "NN", "and": "CC", "patiently": "RB", "waiting": "VBG", "for": "IN", "his": "PRP$", "son": "NN", "to": "TO", "arrive": "VB"
              }
            }
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
        client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

        text = "Delhi has a population of 1.3 crore. Arvind Kejriwal is the Chief Minister of Delhi"

        dltk_ner_tagger = client.ner_tagger(text)

        print(dltk_ner_tagger)

**Output**

.. code-block:: JSON

    {
      "spacy": {"result": {"Delhi": "GPE", "1.3": "CARDINAL", "Arvind Kejriwal": "PERSON"}}
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
        client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

        text = "And now for something completely different."

        dltk_dependency_parser_response = client.dependency_parser(text)

        print(dltk_dependency_parser_response)

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

        import dltk_ai
        client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

        text = "Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality."

        dltk_tags = client.tags(text)

        print(dltk_tags)

**Output**

    .. code-block:: JSON

        {
            "rake": {
                "tags": ["artificial intelligence", "intelligence demonstrated", "machines", "unlike", "natural intelligence displayed", "humans", "animals", "involves consciousness", "emotionality"]
                }  
        }

