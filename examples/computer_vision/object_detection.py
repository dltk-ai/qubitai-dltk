import dltk_ai

client = dltk_ai.DltkAiClient("YOUR_APIKEY")
file_path = "../data/image/object_detection_sample_1.jpg"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg/515px-2018_BMW_X3_xDrive30d_M_Sport_Automatic_3.0_Front.jpg"


# Note: simplest form
response = client.object_detection(image_path=file_path)
print(response)

# Note: using Azure
response = client.object_detection(image_path=file_path, azure=True)
print(response)

# Note: using image_url
response = client.object_detection(image_url=image_url, azure=True)
print(response)

# Note: Avoiding tensorflow based model
response = client.object_detection(image_path=file_path, tensorflow=False, azure=True)
print(response)

# Note: Getting Image as response output
response = client.object_detection(image_path=file_path, tensorflow=True, output_types=["json", "image"])
print(response)
