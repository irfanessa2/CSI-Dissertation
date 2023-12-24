import cv2
import threading
import queue

class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue()
        self.stopped = False

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera stream: {src}")

        # Start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self):
        while True:
            if self.stopped:
                return

            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass

            self.q.put(frame)

    def read(self):
        return self.q.get()

    def more(self):
        return self.q.qsize() > 0

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    camera_url = 'rtsp://irfanessa:Locker123@192.168.1.156/stream1'
    stream = CameraStream(camera_url)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        while True:
            if stream.more():
                frame = stream.read()

                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # Draw green squares around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the captured frame
                cv2.imshow('Tapo Camera Live', frame)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
