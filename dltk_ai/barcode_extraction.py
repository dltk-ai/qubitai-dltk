from pyzbar import pyzbar as pyzbar
import cv2

def extract(image):
    '''
    Extract the serial number or encoded text from the barcode or QR code image 
    Parameters:
        image : cv2 image
    Returns:
        serialno : list of serial number text extracted
        bboxes : list of bounding boxes
        code_type : list of detected codes 
    '''
    image_read = cv2.imread(image)
    response = None
    try:
        decode = pyzbar.decode(image_read) 
        
        if len(decode) < 1:
            return None
        
        response = []
        for code in decode:
            code_detail = {'serial_number': code.data.decode("utf-8"), 'code_type':code.type } 
            x,y,w,h = code.rect
            code_detail['bbox'] = (x, y, w, h)
            response.append(code_detail)

    except Exception as e:
        print("Failed to process image due to ", e)

    return response



