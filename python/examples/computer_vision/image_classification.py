import dltk_ai

client = dltk_ai.DltkAiClient("YOUR_APIKEY")
file_path = "../../img/umbrella.jpg"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg"


# Note: Image classification: Simplest form
response = client.image_classification(image_path=file_path)
print(response)

# Note: top_n=4 predictions, image classification
response = client.image_classification(image_path=file_path, top_n=4)
print(response)

# Note: predictions using ibm & azure
response = client.image_classification(image_path=file_path, azure=True, ibm=True)
print(response)


