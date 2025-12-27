import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
from ultralytics import YOLO


class YoloPublisher(Node):

    def __init__(self):
        super().__init__('yolo_publisher')

        # publisher in ROS 2
        self.publisher_ = self.create_publisher(String, 'detected_fruit', 10)

        # YOLO model
        self.model = YOLO("/home/user/projects/yolo/best.pt")

        # Orbbec camera
        self.cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("YOLO publisher started")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No image from camera")
            return

        results = self.model(frame)
        detections = results[0]

        best_conf = 0.0
        best_class = None

        if detections.boxes is not None:
            for box in detections.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                if conf > best_conf:
                    best_conf = conf
                    best_class = class_name

        if best_class is not None:
            msg = String()
            msg.data = f"{best_class} {best_conf:.2f}"
            self.publisher_.publish(msg)
            self.get_logger().info(f"Publishing: {msg.data}")

        # wizualizacja (opcjonalna)
        annotated = detections.plot()
        cv2.imshow("YOLO + Orbbec", annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
