import dltk_ai


def main():
    c = dltk_ai.DltkAiClient('YOUR_APIKEY')

    # the api extracts images from given urls
    url1 = "url to extract images from"
    url2 = "url to extract images from"

    response = c.visual_search(url1,url2)
    print(response)


if __name__ == '__main__':
    main()
