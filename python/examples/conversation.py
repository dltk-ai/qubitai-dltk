import dltk_ai
import base64


def main():
    c = dltk_ai.DltkAiClient('YOUR_APIKEY')

    # speech_to_text
    # specify the path from where you want to upload the audio file
    path = 'audio1.wav'
    response = c.speech_to_text(path)
    print(response)

    # speech to text compare
    # specify the algothim (google/ibm_watson) you would like to use for converting speech to text
    path = 'audio.wav'
    response = c.speech_to_text_compare(path,'google')
    print(response)


if __name__ == '__main__':
    main()