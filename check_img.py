python

class ImageProcessor:
    def __init__(self, path, train_file):
        self.path = path
        self.train_file = train_file

    def process_images(self):
        result = os.listdir(self.path)
        num = 0
        if not os.path.exists(self.train_file):
            os.mkdir(self.train_file)
        for i in result:
            try:
                image = cv2.imread(self.path + '/' + i)
                cv2.imwrite(self.train_file + '/' + 'Compressed' + i, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                num += 1
            except:
                pass
        print('数据有效性验证完毕,有效图片数量为 %d' % num)
        if num == 0:
            print('您的图片命名有中文，建议统一为1（1）.jpg/png')

