**************
Speech to Text
**************

Extracts text from given audio file.

.. note:: Supports english language

.. function:: client.speech_to_text(audio, sources):

   :param audio: path of audio file to extract text from
   :param sources: list - Algorithm to use for extracting text from audio. Supported sources - google, ibm_watson. Default - google 
   :rtype: A json object containing text returned from respective Algorithm

example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')
   
    speech_to_text_reponse = client.speech_to_text(audio_path, sources=['google','ibm_watson'])
    print(speech_to_text_reponse)