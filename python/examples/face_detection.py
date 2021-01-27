import dltk_ai


def save_img(response, filepath):
    with open(filepath, "wb") as f:
        f.write(response)


def main():
    c = dltk_ai.DltkAiClient('YOUR_APIKEY')

    response = c.face_detection_image('../img/fd-actual-img.jpg')  # it will return image data in bytes.
    print(response)
    save_img(response, "../img/fd-response.png")
    response = c.face_detection_json('../img/fd-actual-img.jpg')  # it will return the co-ordinates of detected faces
    print(response)
    response = c.eye_detection_image('../img/fd-actual-img.jpg')  # it will return image data in bytes.
    print(response)
    save_img(response, "../img/fd-response.png")
    response = c.eye_detection_json('../img/fd-actual-img.jpg')  # it will return the co-ordinates of detected faces
    print(response)

    response = c.face_detection_image_core('../img/fd-actual-img.jpg')  # it will return image data in bytes.
    print(response)
    save_img(response, "../img/fd-response.png")
    response = c.face_detection_json_core(
        '../img/fd-actual-img.jpg')  # it will return the co-ordinates of detected licence plates.
    print(response)

    response = c.object_detection_image('../img/lp-actual-img.jpg')  # it will return image data in bytes.
    print(response)
    save_img(response, "../img/lp-response.png")
    response = c.object_detection_json(
        '../img/lp-actual-img.jpg')  # it will return the co-ordinates of detected licence plates.
    print(response)

    response = c.image_classification('../img/lp-actual-img.jpg')
    print(response)


if __name__ == '__main__':
    main()
