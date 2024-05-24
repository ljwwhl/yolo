from ultralytics import YOLO

# Load a model
model = YOLO("yolov8+MobileNetV3+SE.yaml")  # build a new model from scratch


# Use the model
model.train(data="/root/autodl-tmp/RUOD_yolo/my_data.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format