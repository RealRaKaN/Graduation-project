from ultralytics import YOLO

model = YOLO("runs/detect/train11/weights/best.pt")

model.predict(
    source=0,
    show=True,
    conf=0.90,
    vid_stride=2,
    imgsz=900
)
'''model = YOLO("yolov8.pt")

model.train(
    data="YOLO8-objects-1/data.yaml",
    epochs=50,
    imgsz=640
)
'''