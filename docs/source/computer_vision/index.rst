*****
About
*****

DLTK's Computer Vision service includes advanced image processing algorithms which return required information from the given image. We also have third party APIs like IBM & Azure integrated.


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
   :param top_n: If True, uses dlib for face analyticsget top n predictions
   :param tensorflow: If True, uses tensorflow for image classification
   :param azure: If True, returns azure results of image classification on given image
   :param ibm: If True, uses mtcnn for face analyticsif True, returns ibm results of image classification on given image
   :param list output_types: Type of output requested by client: "json", "image"
   :rtype: Image classification response


**Example**:: 

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    file_path = "../image_classification_sample.jpg"

    # Note: top_n=4 predictions, image classification
    image_classification = client.image_classification(image_path=file_path, top_n=4)

    print(image_classification)

**Input**

.. image:: https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg
    :alt: sample-classification-image
    :height: 200

**Output**

.. code-block:: JSON

    {
      "task_status": "SUCCESS", 
      "task_id": "331348a2-254d-418a-a1b8-9414172f8a86", 
      "output": {
        "tensorflow_predicted_classes": [{"class": "quilt", "confidence": "0.3350585"}, {"class": "tiger_cat", "confidence": "0.16452688"}, {"class": "lynx", "confidence": "0.09777632"}, {"class": "sleeping_bag", "confidence": "0.06928878"}]
        }
    }


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
     - Data Trained on
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
   :param tensorflow: If True, uses tensorflow for object detection
   :param azure: If True, returns azure results of object detection on given image
   :param list output_types: Type of output requested by client: "json" (bounding box coordinates for each object found), "image" (base64 encoded object)
   :rtype: A json object containing the output of object detection

**Example**::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg"

    object_detection_response = client.object_detection(image_url=image_url)

    print(object_detection_response)

**Input**

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg
    :alt: sample-object-image
    :height: 200

**Output**

.. code-block:: JSON

    {
      "task_status": "SUCCESS", 
      "task_id": "b37318f5-c657-4e93-a079-f3888fe03717", 
      "output": {
        "tensorflow_detected_objects": [{"object_name": "car", "confidence": 0.8257747888565063, "bbox": {"x1": 14, "y1": 23, "x2": 501, "y2": 257}}]
        }
    }


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
   :param dlib: If True, uses dlib for face analytics
   :param opencv: If True, uses opencv for face analytics
   :param azure: If True, returns azure results of face analytics on given image
   :param mtcnn: If True, uses mtcnn for face analytics
   :param list output_types: Type of output requested by client: "json", "image"
   :rtype: A json object containing the output of object detection

**Example**::

    import dltk_ai
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    image_url = "https://images.financialexpress.com/2020/01/660-3.jpg"

    face_analytics_response = client.face_analytics(image_url=image_url, azure=True)

    print(face_analytics_response)

**Input**

.. image:: https://images.financialexpress.com/2020/01/660-3.jpg
    :alt: sample-face-image
    :height: 200

**Output**

.. code-block:: JSON

    {
      "task_status": "SUCCESS", 
      "task_id": "4e915abe-33da-4add-a325-8a07cf2093c3", 
      "output": 
        {
          "opencv": {"json": {"face_locations": [{"x": 356, "y": 129, "w": 79, "h": 117}, {"x": 22, "y": 129, "w": 81, "h": 120}, {"x": 123, "y": 134, "w": 77, "h": 109}, {"x": 231, "y": 137, "w": 77, "h": 111}, {"x": 567, "y": 117, "w": 74, "h": 117}, {"x": 460, "y": 126, "w": 76, "h": 123}]}}, 
          "azure": {"json": {"face_locations": [{"x": 27, "y": 153, "w": 87, "h": 87}, {"x": 558, "y": 139, "w": 85, "h": 85}, {"x": 348, "y": 151, "w": 82, "h": 82}, {"x": 231, "y": 158, "w": 79, "h": 79}, {"x": 121, "y": 155, "w": 78, "h": 78}, {"x": 453, "y": 154, "w": 78, "h": 78}]}}
          }
    }


