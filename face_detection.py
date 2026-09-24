import cv2
import mediapipe as mp

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  # Convert frame to grayscale for detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  # Draw rectangle around detected faces
  for x, y, w, h in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  cv2.imshow('BCA AI Demo - Face Detection', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
