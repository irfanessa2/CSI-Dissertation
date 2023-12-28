import cv2
import threading
import queue
import time


class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(1)
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
    #camera_url = "rtsp://192.168.1.88:8554/c920"
    # stream = CameraStream(camera_url)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    prev_frame_time, new_frame_time = 0, 0

    INTERLEAVE = 5
    frame_cnt = 0
    faces = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        new_frame_time = time.time()

        # FPS calculation
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "{:.2f}".format(fps)  # Format FPS to two decimal places

        if frame_cnt < 1:
            frame_cnt = INTERLEAVE
            # Convert frame to grayscale for face detectionw
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in faces:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put FPS text on frame
        cv2.putText(
            orig, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        print(f"{fps:.2f}", end="\r", flush=True)

        # Display the captured frame

        cv2.imshow("C920 Live", orig)
        frame_cnt -= 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
