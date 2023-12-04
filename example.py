python

class YOLOv5Detector:
    def __init__(self, weights, data, device='', half=False, dnn=False):
        self.weights = weights
        self.data = data
        self.device = device
        self.half = half
        self.dnn = dnn
        self.model, self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.load_model()

    def draw_box_string(self, img, box, string):
        x, y, x1, y1 = box
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./simhei.ttf", 20, encoding="utf-8")
        draw.text((x-40, y-25), string, (255, 0, 0), font=font)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def load_model(self):
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'
        if pt or jit:
            model.model.half() if self.half else model.model.float()
        return model, stride, names, pt, jit, onnx, engine

    def run(self, img, imgsz=(640, 640), conf_thres=0.1, iou_thres=0.05, max_det=1000, classes=None,
            agnostic_nms=False, augment=False):
        cal_detect = []
        device = select_device(self.device)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        im = letterbox(img, imgsz, self.stride, self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]}'
                    cal_detect.append([label, xyxy, float(conf)])
        return cal_detect

    def detect(self, image_path):
        image = cv2.imread(image_path)
        results = self.run(self.model, image, self.stride, self.pt)
        for i in results:
            box = i[1]
            if str(i[0]) == 'yumi_aihuayebing':
                i[0] = '玉米矮花叶病'
            if str(i[0]) == 'yumi_huibanbing':
                i[0] = '玉米灰斑病'
            if str(i[0]) == 'yumi_huibanbings':
                i[0] = '玉米灰斑病'
            if str(i[0]) == 'yumi_xiubing':
                i[0] = '玉米锈病'
            if str(i[0]) == 'yumi_xiubings':
                i[0] = '玉米锈病'
            if str(i[0]) == 'yumi_yebanbing':
                i[0] = '玉米叶斑病'
            if str(i[0]) == 'yumi_yebanbings':
                i[0] = '玉米叶斑病'
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            image = self.draw_box_string(image, [int(box[0]), int(box[1]), int(box[2]), int(box[3])], str(i[0]) + ' ' + str(i[2])[:5])
        cv2.imshow('image', image)
        cv2.waitKey(0)

detector = YOLOv5Detector(ROOT / 'best.pt', ROOT / 'data/coco128.yaml')
detector.detect(r"C:\Users\15690\Desktop\2000yolo\code\images\a (369).jpg")
