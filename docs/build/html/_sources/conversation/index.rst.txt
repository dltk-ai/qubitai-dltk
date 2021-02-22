*****
About
*****

DLTK's Conversation service enables enables transcription of audio files into text. We have third party APIs like IBM & Azure integrated.

**************
Speech to Text
**************

Extracts text from given audio file.

*Supported Third Party Classifiers*

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Name
     - Documentation Link
   * - Azure
     - https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-to-text
   * - IBM
     - https://cloud.ibm.com/apidocs/speech-to-text

.. note:: 
    * Supported Languages - English
    * Supported Audio File - .wav


.. function:: client.speech_to_text(audio, sources):

   :param audio: path of audio file to extract text from
   :param sources: list - Algorithm to use for extracting text from audio. Supported sources - google, ibm_watson. Default - google 
   :rtype: A json object containing text returned from respective Algorithm

**Example**::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')
    audio_path = "../recording.wav"
   
    speech_to_text_reponse = client.speech_to_text(audio_path, sources=['google','ibm_watson'])
    print(speech_to_text_reponse)