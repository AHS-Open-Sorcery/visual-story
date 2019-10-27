from imageai.Detection import ObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

execution_path = os.getcwd()
def getObjects(filename):
    modelType = "yolo"
    detector = ObjectDetection()
    if (modelType=="yolo"):
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path, "preTrainedModels/yolo.h5"))
    elif (modelType=="retina"):
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "preTrainedModels/resnet50Coco.h5"))

    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , filename), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

    for eachObject in detections:
        yield (eachObject["name"], eachObject["percentage_probability"])


def getOccupation(filename):
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("preTrainedModels/idenprof.h5")
    prediction.setJsonPath("preTrainedModels/idenprof.json")
    prediction.loadModel(num_objects=10)

    predictions, probabilities = prediction.predictImage(filename, result_count=3)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        yield (eachPrediction, eachProbability)
