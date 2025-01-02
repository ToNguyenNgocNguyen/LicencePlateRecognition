import cv2
import supervision as sv
from ultralytics import YOLO
import re
import numpy as np
from rapidocr_onnxruntime import RapidOCR

TEXT_DICT = {"text": ""}

def text_on_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    color = (255, 255, 0)  # Green text (BGR format)

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    text_x, text_y = 10, 200 + text_height
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)


def is_valid_license_plate(plate):
    # Define the pattern for a Chinese license plate
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"    
    # Match the plate against the pattern
    if re.match(pattern, plate):
        return True
    return False


def load_model():
    license_recognition = RapidOCR()
    license_detector = YOLO("model/LicenseDetection.pt")
    return license_recognition, license_detector


def ocr(image, detections):
    text = ""
    for x_min, y_min, x_max, y_max in detections.xyxy:
        license_plate = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        license_plate = cv2.resize(license_plate, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        license_plate = cv2.GaussianBlur(license_plate, (5,5), 0)
        license_plate = cv2.medianBlur(license_plate, 5)
        ret, license_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        license_plate = cv2.bitwise_not(license_plate)

        text, elapse = license_recognition(license_plate, use_det=False)
        text = text[0][0].replace(" ", "").strip()

        if is_valid_license_plate(text):
            text = text[:2] + " " + text[2:4] + " " + text[4:6] + " " + text[6:]
            TEXT_DICT["text"] = text

        cv2.imshow(f"license plate: {TEXT_DICT["text"]}", license_plate)
        

def detection(image):
    results = license_detector(image, conf=0.5, imgsz=640)[0]
    # print(image.shape)
    polygon = np.array([[0, 520] , [0, 600], [1280, 600], [1280, 520]])
    zone = sv.PolygonZone(polygon=polygon)
    detections = sv.Detections.from_ultralytics(results)
    mask = zone.trigger(detections=detections)
    detections = detections[mask]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN)

    if len(detections.confidence) <= 0:
        TEXT_DICT["text"] = ""

    if TEXT_DICT["text"] == "":
        ocr(image, detections=detections)

    # text = ocr(image, detections=detections)
    # TEXT_DICT["text"] = text
    text_on_image(image, TEXT_DICT["text"])
    
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_image = zone_annotator.annotate(
        scene=image, label="zone")

    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections)

    return annotated_image


if __name__=="__main__":
    license_recognition, license_detector = load_model()
    video_path = "videotest/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            annotated_frame = detection(frame)

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            key = cv2.waitKey(int(delay * 1000))

            if key == 27 or key == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
