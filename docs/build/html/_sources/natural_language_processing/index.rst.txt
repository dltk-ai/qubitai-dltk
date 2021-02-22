*****
About
*****

DLTK's NLP service has pre built Models for various Natural Language Processing tasks, which can be leveraged to predict on your text. We also have third party APIs like IBM & Azure integrated.

******************
Sentiment Analysis
******************

Detects positive or negative sentiment in text.

Usage: social media monitoring, Brand reputation analysis, etc.

.. py:function:: client.sentiment_analysis(text, sources = ['spacy'])

   :param text: text for analysing the sentiment
   :param sources: algorithm to use - azure/ibm_watson/spacy. Default is spacy.
   :rtype: A json object containing the sentiment analysis response

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "The product is very easy to use and has got a really good life expectancy.

    sentiment_analysis_response = client.sentiment_analysis(text)

    print(sentiment_analysis_response)


**********************
Parts of Speech Tagger
**********************

Identifies and marks a word in a text as corresponding to a particular part of speech, based on both its definition and its context.

.. py:function:: client.pos_tagger(text, sources = ['spacy'])

   :param text: text for Parts of Speech Tagging
   :param sources: algorithm to use - ibm_watson/spacy
   :rtype: A json object containing the Parts of Speech of the words in the sentence.

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "They refuse to permit us to obtain the refuse permit.

    pos_tagger_response = client.pos_tagger(text)

    print(pos_tagger_response)


************************
Named Entity Recognition
************************

Identifies key information (entities) in text. Each token is given an appropriate tag such as Person, Location, Organisation, etc.

.. py:function:: client.ner_tagger(text, sources = ['spacy'])

   :param text: text for Named Entity Recognition
   :param sources: algorithm to use - azure/ibm_watson/spacy
   :rtype: A json object with Entities identified in the given text.

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "John has moved to California recently.

    ner_tagger_response = client.ner_tagger(text)

    print(ner_tagger_response)


*****************
Dependancy Parser
*****************

Analyses the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads.

Usage: Grammar monitoring.

.. py:function:: client.dependency_parser(text):

   :param text: text for dependency parser
   :rtype: A json object with Entities identified in the given text.

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "And now for something completely different.

    dependency_parser_response = client.dependency_parser(text)

    print(dependency_parser_response)


****************
Tags Recognition
****************

Identifies the important words in a sentence.

.. py:function:: client.tags(text)

   :param text: text for tags recognotion
   :rtype: A json object with Tags identified in the given text.

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    text = "Elon Musk has shared a photo of the spacesuit designed by SpaceX. This is the second image shared of the new design and the first to feature the spacesuit full-body look.."

    tags_response = client.tags(text)

    print(tags_response)

