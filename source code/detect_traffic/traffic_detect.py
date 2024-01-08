import numpy as np
import supervision as sv
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8s.pt')

# Open video capture and set resolution
cap = cv2.VideoCapture(1)  # Use webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get webcam resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load colors
colors = sv.ColorPalette.default()

# 색상 형식 확인 및 변환 함수
def ensure_color_format(color):
    if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color):
        return color
    else:
        return (255, 0, 0)  # 기본 색상: 빨간색

# Original image resolution
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

# Define multiple sets of polygons for different roads
polygons_road1 = [polygon1_road1, polygon2_road1]
polygons_road2 = [polygon1_road2, polygon2_road2]
polygons_road3 = [polygon1_road3, polygon2_road3]
polygons_road4 = [polygon1_road4, polygon2_road4]

# 현재 활성화된 폴리곤 세트 (기본값으로 road1 설정)
current_polygons = polygons_road1

# 폴리곤 및 박스 표시 여부를 결정하는 플래그 추가
show_polygons = True
show_boxes = True

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

# 폴리곤에 숫자 지우는 함수 정의
def draw_polygon_without_count(scene, polygon, zone_annotator, thickness):
    cv2.polylines(scene, [polygon], isClosed=True, color=zone_annotator.color, thickness=thickness)
    return scene

# process_frame 함수 정의
def process_frame(frame: np.ndarray, model, zones, zone_annotators, box_annotators, show_count=False, show_boxes=False) -> np.ndarray:
    results = model(frame, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Car class ID for filtering (assuming 'car' class ID is 2)
    car_class_id = 2

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        # Car detections 필터링
        car_detections = [d for d, class_id in zip(detections.xyxy, detections.class_id) if class_id == car_class_id]

        # 각 탐지에 대해 mask 계산
        mask = np.array([zone.trigger(detections=sv.Detections(xyxy=np.array([d]), confidence=np.array([detections.confidence[idx]]), class_id=np.array([class_id]))) for idx, (d, class_id) in enumerate(zip(detections.xyxy, detections.class_id)) if class_id == car_class_id])
        detections_filtered = [d for d, m in zip(car_detections, mask) if m]
        object_count = len(detections_filtered)  # 각 zone에 있는 차량 수 계산
        # process_frame 함수 내
        if show_boxes:
            for detection in detections_filtered:
                frame = box_annotator.annotate_box_only(scene=frame, detection=detection)

        # 폴리곤 조건부 주석 처리
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

# 메인 루프 시작 전에 update_annotators 변수 초기화
update_annotators = False

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        break

    # Check for key presses
        # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_polygons = polygons_road1
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('2'):
        current_polygons = polygons_road2
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('3'):
        current_polygons = polygons_road3
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('4'):
        current_polygons = polygons_road4
        show_polygons = True
        show_boxes = True
        update_annotators = True
    elif key == ord('5'):
        current_polygons = []  # 폴리곤을 숨깁니다.
        show_polygons = False
        show_boxes = False
        update_annotators = True
    else:
        update_annotators = False

    if update_annotators:
        # Update current zones and annotators
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

    frame = process_frame(frame, model, current_zones, zone_annotators, box_annotators, show_count=False, show_boxes=show_boxes)

    # Display the frame
    cv2.imshow('Frame', frame)

    # 'q' key to break the loop
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()