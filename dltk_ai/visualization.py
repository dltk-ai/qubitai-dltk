from PIL import Image, ImageDraw, ImageFont
import sys


def get_coordinates_from_bbox(bbox, coords_order):
    """
    This function is for reformatting bbox coordinates for downstream tasks
    Args:
        bbox: bounding box
        coords_order: coordinate order can be either 'xywh' or 'x1y1x2y2' format

    Returns:
        x1,y1,x2,y2
    """
    if coords_order == 'xywh':
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
    elif coords_order == 'x1y1x2y2':
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError("Please choose, coords_order from these 2 ['xywh', 'x1y1x2y2']")
    return x1, y1, x2, y2


def draw_bbox(image, bboxes, captions=[], bbox_color="#FF0066", bbox_thickness=6, font_size=24,
              font_color=(255, 255, 255), coords_order='x1y1x2y2'):
    """
    This function is for drawing bounding boxes on image.

    Args:
        image: Image in PIL format
        bboxes: list of bounding box
        captions: list of caption to be displayed corresponding to each box
        bbox_color: color of bounding box
        bbox_thickness: thickness of bounding box
        font_size: font size of caption
        font_color: font color of caption
        coords_order: order of coordinates can be either 'xywh' or 'x1y1x2y2' format

    Returns:
        Image in PIL format with bounding box
    """
    pil_img = image.copy()

    if len(captions) > 0:
        assert len(bboxes) == len(captions), "Please ensure number of captions is same as number of bounding boxes"

    img = ImageDraw.Draw(pil_img)

    offset = bbox_thickness // 2
    # Draw bbox
    for bbox in bboxes:
        x1, y1, x2, y2 = get_coordinates_from_bbox(bbox, coords_order)
        offset = bbox_thickness // 2
        bbox = [x1 - offset, y1 - offset, x2 + offset, y2 + offset]
        img.rectangle(bbox, outline=bbox_color, width=bbox_thickness)

        offset = 1
        bbox = [x1 - offset, y1 - offset, x2 + offset, y2 + offset]
        img.rectangle(bbox, outline=(255, 255, 255), width=2)

    if len(captions) > 0:

        if sys.platform == 'win32':
            font = ImageFont.truetype("arial.ttf", font_size)

        elif sys.platform == 'linux':
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)

        # Put caption on Image
        for caption, bbox in zip(captions, bboxes):
            # get pixel size font is going to take
            text_width, text_height = font.getsize(caption.capitalize())
            x1, y1, x2, y2 = get_coordinates_from_bbox(bbox, coords_order)
            bbox = [x1 + offset, y1 + offset, x1 + offset + text_width, y1 + offset + text_height + text_height // 10]
            img.rectangle(bbox, fill=bbox_color, outline=bbox_color, width=bbox_thickness)
            img.text((x1 + offset, y1 + offset), caption.capitalize(), font=font, fill=font_color)

    return pil_img
