python
class YOLOv5:
    def __init__(self, weights, imgsz=(640, 640)):
        self.weights = weights
        self.imgsz = imgsz
        self.model = self._build_model()

    def _build_model(self):
        # Load model architecture and weights
        model = ...
        model.load_weights(self.weights)
        return model

    def detect(self, image):
        # Preprocess image
        image = self._preprocess_image(image)

        # Run inference
        output = self.model.predict(image)

        # Postprocess output
        detections = self._postprocess_output(output)

        return detections

    def _preprocess_image(self, image):
        # Resize image
        image = cv2.resize(image, self.imgsz)

        # Normalize image
        image = image / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _postprocess_output(self, output):
        # Process output to get detections
        detections = ...

        return detections
