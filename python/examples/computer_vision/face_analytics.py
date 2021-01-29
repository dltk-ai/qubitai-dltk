import dltk_ai

dltkai = dltk_ai.DltkAiClient('YOUR API-KEY')
response = dltkai.face_analytics(image_path='../../img/fd-actual-img.jpg',
                                 output_types=["json"])
