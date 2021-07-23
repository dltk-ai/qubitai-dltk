from PIL import Image, ImageDraw, ImageFont
import sys
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def model_evaluation_plots(task, data, plots):

    """

    Args:
        task: classification/regression
        data: Output response from training job
        plots: List of plots

    Returns:
        plots
    """

    # convert plot to list if string is given
    if type(plots) == str:
        plots = [plots]

    # check service names
    allowed_service = ['regression', 'classification']
    task = task.lower()
    assert task in allowed_service, f"Please select *service* from {allowed_service}"

    # allowed plots
    allowed_plots = ["residual_plot"] if task=="regression" else ["prediction_distribution","cumulative_gains","lift_chart", "roc_curve","confusion_matrix", "ks_metric_plot"]
    for plot in plots:
        assert plot in allowed_plots, "please choose from {}".format(allowed_plots)

    data = data['output']['eval']

    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    axes = {}

    for plot_number in range(len(plots)):
        axes[plots[plot_number]] = fig.add_subplot(spec[plot_number])
    if task == "classification":
        # class names
        classes = list(data['prediction_distribution'].keys())
        # prediction distribution
        if "prediction_distribution" in plots:
            sns.kdeplot(x=data['prediction_distribution'][classes[0]], color='g', fill=True, label=classes[0],
                        ax=axes["prediction_distribution"])
            sns.kdeplot(x=data['prediction_distribution'][classes[1]], color='r', fill=True, label=classes[1],
                        ax=axes["prediction_distribution"])
            axes["prediction_distribution"].set_xlabel("probability")
            axes["prediction_distribution"].set_title("probability density")

        # cumulative gains chart
        if "cumulative_gains" in plots:
            sns.lineplot(x=data['cumulative_gains'][classes[0]]['percentage'],
                         y=data['cumulative_gains'][classes[0]]['gains'], label=classes[0], ax=axes["cumulative_gains"])
            sns.lineplot(x=data['cumulative_gains'][classes[1]]['percentage'],
                         y=data['cumulative_gains'][classes[1]]['gains'], label=classes[1], ax=axes["cumulative_gains"])
            axes["cumulative_gains"].set_xlabel("probability")
            axes["cumulative_gains"].set_ylabel("gains")
            axes["cumulative_gains"].set_title("cumulative gains")

        # lift chart
        if "lift_chart" in plots:
            sns.lineplot(x=data['lift_chart'][classes[0]]['percentage'], y=data['lift_chart'][classes[0]]['gains'],
                         label=classes[0], ax=axes["lift_chart"])
            sns.lineplot(x=data['lift_chart'][classes[1]]['percentage'], y=data['lift_chart'][classes[1]]['gains'],
                         label=classes[1], ax=axes["lift_chart"])
            axes["lift_chart"].set_xlabel("probability")
            axes["lift_chart"].set_ylabel("gains")
            axes["lift_chart"].set_title("lift")

        # roc curve
        if "roc_curve" in plots:
            roc_values = data['rocCurve']
            roc_df = pd.DataFrame(roc_values, columns=['tpr', 'fpr'])
            sns.scatterplot(data=roc_df, x=roc_df['fpr'], y=roc_df['tpr'], ax=axes["roc_curve"])
            axes["roc_curve"].set_title("ROC Curve")

        # confusion matrix
        if "confusion_matrix" in plots:
            confusion_matrix_list = data['confusionMatrix']
            sns.heatmap(confusion_matrix_list, fmt='g', annot=True, ax=axes["confusion_matrix"])
            axes["confusion_matrix"].set_title("Confusion Matrix")

        # ks metric
        if "ks_metric_plot" in plots:
            sns.lineplot(x=data["ks_metric_values"][classes[0]]['thresholds'],
                         y=data["ks_metric_values"][classes[0]]['y_axis'], label=classes[0], ax=axes["ks_metric_plot"])
            sns.lineplot(x=data["ks_metric_values"][classes[1]]['thresholds'],
                         y=data["ks_metric_values"][classes[1]]['y_axis'], label=classes[1], ax=axes["ks_metric_plot"])
            axes["ks_metric_plot"].legend(loc='lower right')
            axes["ks_metric_plot"].text(0.7, 0.2,
                                        "ks_statistic = {}".format(round(data["ks_metric_values"]["ks_statistic"], 2)))
            axes["ks_metric_plot"].text(0.7, 0.3, "max_distance = {}".format(
                round(data["ks_metric_values"]["max_distance_at"], 2)))
            axes["ks_metric_plot"].set_xlabel("thresholds")
            axes["ks_metric_plot"].set_title("KS Metric")

    elif task == "regression":
        if "residual_plot" in plots:
            sns.kdeplot(x=data['residuals'], color='g', ax=axes["residual_plot"])
            axes["residual_plot"].set_xlabel("residuals")
            axes["residual_plot"].set_title("Residual Plot")
