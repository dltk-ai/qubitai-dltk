import dltk_ai


client = dltk_ai.DltkAiClient('YOUR_APIKEY')
# specify the path from where you want to upload the audio file
path = '../data/audio/harvard.wav'

# speech_to_text - Using google
response = client.speech_to_text(path)
print(response)

# speech_to_text - using ibm_watson
response = client.speech_to_text(path,sources=['ibm_watson'])
print(response)

# speech_to_text - using google & ibm_watson
response = client.speech_to_text(path,sources=['ibm_watson','google'])
print(response)


