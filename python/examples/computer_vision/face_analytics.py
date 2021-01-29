import dltk_ai

dltkai = dltk_ai.DltkAiClient('YOUR API-KEY')
response = dltkai.face_analytics(image_path='../data/image/face_analytics_sample_1.jpg',
                                 output_types=["json"])
