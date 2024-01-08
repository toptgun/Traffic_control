import numpy as np
import supervision as sv
import cv2
from ultralytics import YOLO

# Load YOLO models
model = YOLO('yolov8s.pt')
model_accident = YOLO('yolox.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

colors = sv.ColorPalette.default()

def ensure_color_format(color):
    if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color):
        return color
    else:
        return (255, 0, 0)

original_width, original_height = 1796, 1006

# Define polygons for each road
# road1 polygon1 
polygon1_road1 = np.array([
    [int(725 / original_width * frame_width), int(210 / original_height * frame_height)],
    [int(852 / original_width * frame_width), int(210 / original_height * frame_height)],
    [int(676 / original_width * frame_width), int(943 / original_height * frame_height)],
    [int(52 / original_width * frame_width), int(856 / original_height * frame_height)],
    [int(721 / original_width * frame_width), int(213 / original_height * frame_height)]
], dtype=np.int32)

# road1 polygon2 
polygon2_road1 = np.array([
    [int(898 / original_width * frame_width), int(193 / original_height * frame_height)],
    [int(1007 / original_width * frame_width), int(191 / original_height * frame_height)],
    [int(1774 / original_width * frame_width), int(696 / original_height * frame_height)],
    [int(1072 / original_width * frame_width), int(918 / original_height * frame_height)],
    [int(900 / original_width * frame_width), int(202 / original_height * frame_height)]
], dtype=np.int32)

# ROAD2의 polygon1 수정
polygon2_road2 = np.array([
    [int(742 / 1794 * frame_width), int(416 / 1006 * frame_height)],
    [int(1052 / 1794 * frame_width), int(412 / 1006 * frame_height)],
    [int(1352 / 1794 * frame_width), int(930 / 1006 * frame_height)],
    [int(387 / 1794 * frame_width), int(941 / 1006 * frame_height)],
    [int(740 / 1794 * frame_width), int(418 / 1006 * frame_height)]
], dtype=np.int32)

# ROAD2의 polygon2 수정
polygon1_road2 = np.array([
    [int(547 / 1794 * frame_width), int(411 / 1006 * frame_height)],
    [int(715 / 1794 * frame_width), int(419 / 1006 * frame_height)],
    [int(329 / 1794 * frame_width), int(936 / 1006 * frame_height)],
    [int(12 / 1794 * frame_width), int(800 / 1006 * frame_height)],
    [int(546 / 1794 * frame_width), int(410 / 1006 * frame_height)]
], dtype=np.int32)

# ROAD3의 polygon1 수정
polygon2_road3 = np.array([
    [int(841 / 1796 * frame_width), int(338 / 1006 * frame_height)],
    [int(1168 / 1796 * frame_width), int(335 / 1006 * frame_height)],
    [int(1759 / 1796 * frame_width), int(892 / 1006 * frame_height)],
    [int(776 / 1796 * frame_width), int(905 / 1006 * frame_height)],
    [int(842 / 1796 * frame_width), int(347 / 1006 * frame_height)]
], dtype=np.int32)

# ROAD3의 polygon2 수정
polygon1_road3 = np.array([
    [int(486 / 1796 * frame_width), int(319 / 1006 * frame_height)],
    [int(814 / 1796 * frame_width), int(324 / 1006 * frame_height)],
    [int(694 / 1796 * frame_width), int(912 / 1006 * frame_height)],
    [int(14 / 1796 * frame_width), int(817 / 1006 * frame_height)],
    [int(488 / 1796 * frame_width), int(316 / 1006 * frame_height)]
], dtype=np.int32)

# road4의 새로운 polygon1 수정
polygon1_road4 = np.array([
    [int(742 / original_width * frame_width), int(201 / original_height * frame_height)],
    [int(860 / original_width * frame_width), int(199 / original_height * frame_height)],
    [int(727 / original_width * frame_width), int(786 / original_height * frame_height)],
    [int(15 / original_width * frame_width), int(678 / original_height * frame_height)],
    [int(741 / original_width * frame_width), int(202 / original_height * frame_height)]
], dtype=np.int32)

# road4의 새로운 polygon2 수정
polygon2_road4 = np.array([
    [int(895 / original_width * frame_width), int(190 / original_height * frame_height)],
    [int(983 / original_width * frame_width), int(178 / original_height * frame_height)],
    [int(1678 / original_width * frame_width), int(607 / original_height * frame_height)],
    [int(999 / original_width * frame_width), int(671 / original_height * frame_height)],
    [int(897 / original_width * frame_width), int(187 / original_height * frame_height)]
], dtype=np.int32)

polygons_road1 = [polygon1_road1, polygon2_road1]
polygons_road2 = [polygon1_road2, polygon2_road2]
polygons_road3 = [polygon1_road3, polygon2_road3]
polygons_road4 = [polygon1_road4, polygon2_road4]

current_polygons = polygons_road1
current_model = model
show_polygons = True
show_boxes = True
detect_accidents = False
# Convert current polygons to PolygonZone objects
current_zones = [
    sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame_width, frame_height))
    for polygon in current_polygons
]

# Initialize zone annotators and box annotators for both polygons
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=ensure_color_format(colors.by_idx(index % 8)),
        thickness=4,
        text_thickness=8,
        text_scale=4
    )
    for index, zone in enumerate(current_zones)
]

# BoxAnnotator 클래스 정의
class BoxAnnotator(sv.BoxAnnotator):
    def __init__(self, color, thickness=2, text_thickness=1, text_scale=1):
        self.color = ensure_color_format(color)
        self.thickness = thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale

    def annotate_box_only(self, scene, detection):
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(scene, (x1, y1), (x2, y2), self.color, self.thickness)
        return scene

# box_annotators 리스트 초기화
box_annotators = [
    BoxAnnotator(
        color=ensure_color_format(colors.by_idx(index % 8)),
        thickness=4,
        text_thickness=4,
        text_scale=2
    )
    for index, _ in enumerate(current_polygons)
]

def draw_polygon_without_count(scene, polygon, zone_annotator, thickness):
    cv2.polylines(scene, [polygon], isClosed=True, color=zone_annotator.color, thickness=thickness)
    return scene

def process_frame(frame: np.ndarray, model, model_accident, zones, zone_annotators, box_annotators, show_count=False, show_boxes=False, detect_accidents=False) -> np.ndarray:
    results = model(frame, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    detections_accident = []  # Initialize as an empty list or suitable default
    
    if detect_accidents:
        results_accident = model_accident(frame, imgsz=640, verbose=False)[0]
        detections_accident = sv.Detections.from_ultralytics(results_accident)
    
    car_class_id = 2

    for detection in detections_accident:
        bbox = detection[0]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        label = detection[3]
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        if label == 2:
            cv2.putText(frame, "Severe Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif label == 1:
            cv2.putText(frame, "Moderate Accident Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif label == 0:
            cv2.putText(frame, "Fall Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        car_detections = [d for d, class_id in zip(detections.xyxy, detections.class_id) if class_id == car_class_id]
        mask = np.array([zone.trigger(detections=sv.Detections(xyxy=np.array([d]), confidence=np.array([detections.confidence[idx]]), class_id=np.array([class_id]))) for idx, (d, class_id) in enumerate(zip(detections.xyxy, detections.class_id)) if class_id == car_class_id])
        detections_filtered = [d for d, m in zip(car_detections, mask) if m]
        object_count = len(detections_filtered)
        
        if show_boxes:
            for detection in detections_filtered:
                frame = box_annotator.annotate_box_only(scene=frame, detection=detection)

        if show_polygons:
            if show_count:
                frame = zone_annotator.annotate(scene=frame)
            else:
                frame = draw_polygon_without_count(
                    scene=frame,
                    polygon=zone.polygon,
                    zone_annotator=zone_annotator,
                    thickness=zone_annotator.thickness
                )
        cv2.putText(frame, f"Car Count: {object_count}", (int(zone.polygon[0][0]-50), int(zone.polygon[0][1]) + 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    return frame

update_annotators = False

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_polygons = polygons_road1
        show_polygons = True
        show_boxes = True
        update_annotators = True
        detect_accidents = False  # 사고 감지 비활성화
    elif key == ord('2'):
        current_polygons = polygons_road2
        show_polygons = True
        show_boxes = True
        update_annotators = True
        detect_accidents = False  # 사고 감지 비활성화  
    elif key == ord('3'):
        current_polygons = polygons_road3
        show_polygons = True
        show_boxes = True
        update_annotators = True
        detect_accidents = False  # 사고 감지 비활성화
    elif key == ord('4'):
        current_polygons = polygons_road4
        show_polygons = True
        show_boxes = True
        update_annotators = True
        detect_accidents = False  # 사고 감지 비활성화
    elif key == ord('5'):
        current_polygons = []
        show_polygons = False
        show_boxes = False
        update_annotators = True
        detect_accidents = False  # 사고 감지 비활성화
    elif key == ord('6'):
        current_polygons = []
        show_polygons = False
        show_boxes = False
        update_annotators = True
        detect_accidents = not detect_accidents  # 사고 감지 토글

    if update_annotators:
        current_zones = [
            sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame_width, frame_height))
            for polygon in current_polygons
        ]
        zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=ensure_color_format(colors.by_idx(index % 8)),
                thickness=4,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(current_zones)
        ]
        box_annotators = [
            BoxAnnotator(
                color=ensure_color_format(colors.by_idx(index % 8)),
                thickness=4,
                text_thickness=4,
                text_scale=2
            )
            for index, _ in enumerate(current_polygons)
        ]
        update_annotators = False

    frame = process_frame(frame, model, model_accident, current_zones, zone_annotators, box_annotators, show_count=False, show_boxes=show_boxes, detect_accidents=detect_accidents)

    cv2.imshow('Frame', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()