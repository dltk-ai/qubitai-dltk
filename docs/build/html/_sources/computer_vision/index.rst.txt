****************
Object Detection
****************

Identifies and locates objects in an Image.

Usage: Tracking objects, Counting people, Vehicle Detection, etc.



.. function:: client.object_detection(image_url=None, image_path=None, tensorflow=True,
                                    azure=False, output_types=["json"]):


   :param image_url: Image URL
   :param image_path: Local Image Path
   :param tensorflow: if True, uses tensorflow for object detection
   :param azure: if True, returns azure results of object detection on given image
   :param output_types: Type of output requested by client: "json", "image"
   :rtype: A json obj containing the output of object detection
    """
Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg"

    object_detection_response = client.object_detection(image_url=image_url, azure=True)

    print(object_detection_response)


**************
Face Detection
**************

Analyses an individual's face in an image.

Usage: Automated identity verification.

.. note::
    Presently supports Face Detection only.

.. function:: client.face_analytics(image_url=None, features=None, image_path=None, dlib=False,
                                    opencv=True,azure=False, mtcnn=False,output_types=["json"]):


   :param image_url: Image URL
   :param features: Type of features requested by client. Presently supports "face_detection".
   :param image_path: Local Image Path
   :param dlib: if True, uses dlib for face analytics
   :param opencv: if True, uses opencv for face analytics
   :param azure: if True, returns azure results of face analytics on given image
   :param mtcnn: if True, uses mtcnn for face analytics
   :param output_types (list): Type of output requested by client: "json", "image"
   :rtype: A json obj containing the output of object detection

Example::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg"

    face_analytics_response = client.face_analytics(image_url=image_url, azure=True)

    print(face_analytics_response)


********************
Image Classification
********************

Classifies an image according to its visual content.

Usage: Identifying the category of the image, Image organisation, etc,

.. function:: client.image_classification(image_url=None, image_path=None, top_n=3, tensorflow=True,
                                        azure=False, ibm=False,output_types=["json"]):

   :param image_url: Image URL
   :param image_path: Local Image Path
   :param top_n: if True, uses dlib for face analyticsget top n predictions
   :param tensorflow: if True, uses tensorflow for image classification
   :param azure: if True, returns azure results of image classification on given image
   :param ibm: if True, uses mtcnn for face analyticsif True, returns ibm results of image classification on given image
   :param output_types (list): Type of output requested by client: "json", "image"
   :rtype: Image classification response


Example:: 

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    file_path = "../image_classification_sample.jpg"

    # Note: top_n=4 predictions, image classification
    image_classification = client.image_classification(image_path=file_path, top_n=4)

    print(image_classification)

