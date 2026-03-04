
import cv2
import math
import winsound
import threading
from ultralytics import YOLO
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# -------------------- Camera --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------------- Models --------------------
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# 🔥 More accurate model
model = YOLO("yolov8s.pt")

# -------------------- Constants --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22
FRAME_LIMIT = 8   # Faster drowsiness detection

closed_frames = 0
alarm_active = False

# Phone tracking
tracker = None
tracking_phone = False

frame_count = 0


# -------------------- Helper Functions --------------------
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_pts):
    v1 = dist(eye_pts[1], eye_pts[5])
    v2 = dist(eye_pts[2], eye_pts[4])
    h = dist(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h)


def continuous_alarm():
    global alarm_active
    while alarm_active:
        winsound.Beep(1000, 400)


# -------------------- Main Loop --------------------
cv2.namedWindow("Driver Monitor")

while True:

    if cv2.getWindowProperty("Driver Monitor",
                             cv2.WND_PROP_VISIBLE) < 1:
        break

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------- Face & Eye --------------------
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]

        cv2.rectangle(frame,
                      (min(xs), min(ys)),
                      (max(xs), max(ys)),
                      (255, 0, 0), 2)

        left_eye = [(int(landmarks[i].x*w),
                     int(landmarks[i].y*h)) for i in LEFT_EYE]

        right_eye = [(int(landmarks[i].x*w),
                      int(landmarks[i].y*h)) for i in RIGHT_EYE]

        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        EAR = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2.0

        if EAR < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames > FRAME_LIMIT:
            cv2.putText(frame, "DROWSY ALERT!",
                        (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

            if not alarm_active:
                alarm_active = True
                threading.Thread(target=continuous_alarm,
                                 daemon=True).start()
        else:
            alarm_active = False

        cv2.putText(frame, f"EAR: {EAR:.2f}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    # -------------------- Phone Detection + Revalidation --------------------

    # Every 8 frames re-check using YOLO
    if frame_count % 8 == 0:
        results_yolo = model(frame, conf=0.5, verbose=False)

        phone_found = False

        for r in results_yolo:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == "cell phone":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                    tracking_phone = True
                    phone_found = True
                    break

        if not phone_found:
            tracking_phone = False

    # ---- Tracker Update ----
    if tracking_phone:
        success, box = tracker.update(frame)

        if success:
            x, y, w_box, h_box = map(int, box)

            if w_box > 40 and h_box > 40:
                cv2.rectangle(frame,
                              (x, y),
                              (x + w_box, y + h_box),
                              (0, 0, 255), 3)

                cv2.putText(frame,
                            "PHONE DISTRACTION!",
                            (40, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)
            else:
                tracking_phone = False
        else:
            tracking_phone = False

    frame_count += 1

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()