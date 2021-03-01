*****
About
*****

DLTK's Computer Vision service includes advanced image processing algorithms which return required information from the given image. We also have third party APIs like IBM & Azure integrated.

****************
Object Detection
****************

Identifies and locates objects in an Image.

Usage: Tracking objects, Counting people, Vehicle Detection, etc.

*Supported Open Source Models*

.. list-table:: 
   :widths: 25 25 25
   :header-rows: 1

   * - Model
     - Feature Extractor     
     - Pretrained Models
   * - Single Shot Detector
     - ResNet50
     - https://cocodataset.org/#explore

*Supported Third Party Classifiers*

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Name
     - Documentation Link
   * - Azure
     - https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-object-detection


.. function:: client.object_detection(image_url=None, image_path=None, tensorflow=True,
                                    azure=False, output_types=["json"]):


   :param image_url: Image URL
   :param image_path: Local Image Path
   :param tensorflow: if True, uses tensorflow for object detection
   :param azure: if True, returns azure results of object detection on given image
   :param output_types: Type of output requested by client: "json" (bounding box coordinates for each object found), "image" (base64 encoded object)
   :rtype: A json obj containing the output of object detection

**Example**::

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

*Supported Open Source Models*

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Model
     - Implementations
   * - MTCNN
     - https://github.com/ipazc/mtcnn
   * - DLIB-HoG
     - http://dlib.net/python/index.html#dlib.get_frontal_face_detector
   * - OpenCV - DNN
     - https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

*Supported Third Party Classifiers*

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Name
     - Documentation Link
   * - Azure
     - https://docs.microsoft.com/en-us/azure/cognitive-services/face/concepts/face-detection


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

**Example**::

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


*Supported Open Source Models*

.. list-table:: 
   :widths: 25 25 25
   :header-rows: 1

   * - Model
     - Implementations
     - Classes
   * - ResNet50
     - http://www.image-net.org/
     - https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json

*Supported Third Party Classifiers*

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Name
     - Documentation Link
   * - Azure
     - https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-tagging-images
   * - IBM
     - https://cloud.ibm.com/apidocs/visual-recognition/visual-recognition-v3?code=python#getclassify

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


**Example**:: 

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    file_path = "../image_classification_sample.jpg"

    # Note: top_n=4 predictions, image classification
    image_classification = client.image_classification(image_path=file_path, top_n=4)

    print(image_classification)


****************
Barcode Detection
****************

Detects and extracts barcode or QRcode information in an Image.

Usage: Tracking product information and details


.. function:: client.object_detection(image):

   :param image: Local Image Path or Numpy array of the image
   :rtype serial_number: list of serial number text extracted
   :rtype bboxes: list of bounding boxes
   :rtype code_type: list of detected codes 


**Example**::

    import dltk_ai

    response = dltk_ai.barcode_extractor('../barcode.png')
    print(response)

    

