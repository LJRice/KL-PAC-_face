import cv2

# Get user supplied values
imagePath = r"F:\try\1.jpg"  # 保存图片的地址
cascPath = "haarcascade_frontalface_default.xml"
# cascPath = "haarcascade_eye_tree_eyeglasses.xml"
def detect_face(img):

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    # image = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img,faces

if __name__ == '__main__':
    # 打开摄像头，从摄像头中取得视频
    cap = cv2.VideoCapture(0)
    # 定义编码器
    fourcc = cv2.VideoWriter_fourcc(*'mjpg')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        # 读取摄像头
        ret,frame = cap.read()
        # print(ret)
        # 输出当前帧
        # out.write(frame)
        if ret:
            image,faces_coordinate = detect_face(frame)
            cv2.imshow('Video', image)

            # 按下 P 是拍照，O 是退出
            if cv2.waitKey(1) == ord('p'):
                cv2.imwrite(imagePath,image)
                cv2.imshow('saved_image',image)
                while cv2.waitKey(1) != ord('o'):
                    pass
                cv2.destroyWindow('saved_image')
                print('保存成功')

            # 按键盘 Q 退出
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print('No Video')
            break
    out.release()
    cap.release()
    cv2.destroyWindow('Video')

