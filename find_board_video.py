import cv2
from find_board import detect_chessboard

cap = cv2.VideoCapture(1)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open video stream")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = detect_chessboard(frame, video_mode=True, display_output=True, display_debug=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()