import cv2

COMPENSATE_HEIGHT_TSHIRT = .15

class TShirt:
    def __init__(self):
        self.TSHIRT_COORDS = {
            'NECK': 0,
            'LEFT_SHOULDER': 0,
            'RIGHT_SHOULDER': 0,
            'LEFT_HIP': 0,
            'RIGHT_HIP': 0
        }
        self.raw_tshirt_image = None
        self.resized_tshirt = None
        self.contours = None

    def calculate_coords(self, image_path):
        self.raw_tshirt_image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(self.raw_tshirt_image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (3,3), 0)
        thresh = cv2.threshold(blur_image, 220, 255, cv2.THRESH_BINARY_INV)[1]

        self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(self.raw_tshirt_image, self.contours, -1, (0,255,0), 3)

        c = max(self.contours, key=cv2.contourArea)

        # Obtain outer coordinates
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])

        middle_x = ((right[0]-left[0]) // 2) + left[0]
        hip_left_x = (((middle_x-left[0]) // 3) * 2) + left[0]
        hip_right_x = ((right[0]-middle_x) // 3) + middle_x

        shoulder_left_x = (((middle_x-left[0]) // 2)) + left[0]
        shoulder_right_x = ((right[0]-middle_x) // 2) + middle_x

        tshirt_size = bottom[1] - top[1]
        neck_top = int(top[1] + tshirt_size*(COMPENSATE_HEIGHT_TSHIRT/2))
        shoulder_top = int(top[1] + tshirt_size*COMPENSATE_HEIGHT_TSHIRT)

        self.TSHIRT_COORDS['NECK'] = (middle_x, neck_top)
        self.TSHIRT_COORDS['LEFT_SHOULDER'] = (shoulder_left_x, shoulder_top)
        self.TSHIRT_COORDS['RIGHT_SHOULDER'] = (shoulder_right_x, shoulder_top)
        self.TSHIRT_COORDS['LEFT_HIP'] = (hip_left_x, bottom[1])
        self.TSHIRT_COORDS['RIGHT_HIP'] = (hip_right_x, bottom[1])

        """
        # Block for debugging
        cv2.drawContours(raw_image, [c], -1, (36, 255, 12), 2)
        cv2.circle(raw_image, self.TSHIRT_COORDS['NECK'], 8, (255, 255, 0), -1)
        cv2.circle(raw_image, self.TSHIRT_COORDS['LEFT_SHOULDER'], 8, (255, 255, 0), -1)
        cv2.circle(raw_image, self.TSHIRT_COORDS['RIGHT_SHOULDER'], 8, (255, 255, 0), -1)
        cv2.circle(raw_image, self.TSHIRT_COORDS['LEFT_HIP'], 8, (255, 255, 0), -1)
        cv2.circle(raw_image, self.TSHIRT_COORDS['RIGHT_HIP'], 8, (255, 255, 0), -1)
        
        cv2.imshow("tshirt with coords marked", self.raw_tshirt_image)
        cv2.waitKey(0)
        """

    def resize_tshirt(self, HUMAN_COORDS):
        tshirt_size_x = self.TSHIRT_COORDS['RIGHT_SHOULDER'][0] - self.TSHIRT_COORDS['LEFT_SHOULDER'][0]
        human_tshirt_size_x = HUMAN_COORDS['RIGHT_SHOULDER'][0] - HUMAN_COORDS['LEFT_SHOULDER'][0]

        tshirt_size_y = self.TSHIRT_COORDS['LEFT_HIP'][1] - self.TSHIRT_COORDS['LEFT_SHOULDER'][1]
        human_tshirt_size_y = HUMAN_COORDS['LEFT_HIP'][1] - HUMAN_COORDS['LEFT_SHOULDER'][1]


        if tshirt_size_x == 0: tshirt_size_x = 0.001
        if tshirt_size_y == 0: tshirt_size_y = 0.001

        x_factor = (human_tshirt_size_x*100) / tshirt_size_x
        x_factor = round(x_factor/100, 2)

        y_factor = (human_tshirt_size_y*100) / tshirt_size_y
        y_factor = round(y_factor/100, 2)

        # resize image
        self.resized_tshirt = cv2.resize(self.raw_tshirt_image,None,fx=x_factor, fy=y_factor, interpolation = cv2.INTER_CUBIC)

    def get_resized_tshirt(self):
        return self.resized_tshirt

    def update_tshirt_coords(self, resized_tshirt_image):
        gray_image = cv2.cvtColor(resized_tshirt_image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (3,3), 0)
        thresh = cv2.threshold(blur_image, 220, 255, cv2.THRESH_BINARY_INV)[1]

        self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = max(self.contours, key=cv2.contourArea)

        # Obtain outer coordinates
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])

        middle_x = ((right[0]-left[0]) // 2) + left[0]
        hip_left_x = (((middle_x-left[0]) // 3) * 2) + left[0]
        hip_right_x = ((right[0]-middle_x) // 3) + middle_x

        shoulder_left_x = (((middle_x-left[0]) // 2)) + left[0]
        shoulder_right_x = ((right[0]-middle_x) // 2) + middle_x

        shirt_size = bottom[1] - top[1]
        neck_top = int(top[1] + shirt_size*0.075)
        shoulder_top = int(top[1] + shirt_size*0.15)

        self.TSHIRT_COORDS['NECK'] = (middle_x, neck_top)
        self.TSHIRT_COORDS['LEFT_SHOULDER'] = (shoulder_left_x, shoulder_top)
        self.TSHIRT_COORDS['RIGHT_SHOULDER'] = (shoulder_right_x, shoulder_top)
        self.TSHIRT_COORDS['LEFT_HIP'] = (hip_left_x, bottom[1])
        self.TSHIRT_COORDS['RIGHT_HIP'] = (hip_right_x, bottom[1])
