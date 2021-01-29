import dltk_ai

dltkai = dltk_ai.DltkAiClient('552fd7cc-6aed-411d-96d3-2d02a62d375b')
response = dltkai.face_analytics(image_path='../../img/fd-actual-img.jpg', dlib=False, opencv=True, azure=False,
                                 mtcnn=False,
                                 output_types=["json"])
