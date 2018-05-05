from img_process import detect_face as gface
import cv2
import numpy as np
from PIL import Image
import math
import random
import matplotlib.pylab as plt

# detect_face 会检测图片中的人脸位置，并返回一张标记好人脸位置的图片以及各自的坐标
# 预处理：把图片中的人脸都裁剪下来，并统一处理为32*32的大小

eye_cascPath = 'haarcascade_eye.xml'
cut_row = 32
cut_col = 32
# cut_row=112
# cut_col=92
save_f = 0


# 对一张人脸图片裁剪，最终返回 32*32大小的图片
def img_cut(face):
    # rows, cols = face.shape
    # Create the haar cascade
    # print('face = ',face)
    # eye_locs 人眼中心坐标 [x1,y1,x2,y2]
    get_eyes = 0
    eye_locs = []
    while len(eye_locs) != 4:
        eye_locs = []
        get_eyes += 1
        eye_Cascade = cv2.CascadeClassifier(eye_cascPath)
        # Detect eyes in the face
        eyes = eye_Cascade.detectMultiScale(
            # gray,
            face,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(4, 4)
        )
        for (x, y, w, h) in eyes[:2]:
            # cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eye_locs.append(x + w / 2)
            eye_locs.append(y + h / 2)
            tmp_width = w
        # 只识别到一只眼睛，则用几何法决定眼睛
        if len(eye_locs) == 2:
            # print('width=',tmp_width)
            if eye_locs[0] > 30:
                tmp_eye = eye_locs[0]-tmp_width
            else:
                tmp_eye = eye_locs[0]+tmp_width
            eye_locs.append(tmp_eye)
            eye_locs.append(eye_locs[1])
        elif len(eye_locs)==0:
            eye_locs.append(28.5)
            eye_locs.append(55.5)
            eye_locs.append(63)
            eye_locs.append(55.5)
        # print('识别到眼睛坐标为：', eyes)
        if get_eyes == 10:
            break
    print('最终决定眼睛坐标:', eye_locs)
    dx = math.fabs(eye_locs[0] - eye_locs[2])
    dy = math.fabs(eye_locs[1] - eye_locs[3])
    angle = math.atan(dy / dx) * 180 / math.pi
    # 2种情况，①左边眼比右边眼高，②右边眼比左边眼高
    # 左边眼高时
    if eye_locs[1] < eye_locs[3]:
        angle = angle
    # 右边眼高时
    else:
        angle = 360 - angle
    # print('旋转角度为：',angle)
    # 旋转图片，使两眼的中心点处于同一水平线
    # M 为 图片旋转的变换矩阵
    M = cv2.getRotationMatrix2D((eye_locs[0], eye_locs[1]), angle, 1)
    face = cv2.warpAffine(face, M, (200, 200))
    # 裁剪图片，选取人的脸部
    d = int(math.sqrt(dx * dx + dy * dy))
    width = d * 2
    height = d * 2
    fx = abs(int(min(eye_locs[2], eye_locs[0]) - d / 2))
    fy = abs(int(min(eye_locs[1], eye_locs[3]) - d / 2))
    # print('fx,fy,width,height',fx,fy,fx+width,fy+height)
    # 切割图片， 高度,宽度
    face = face[fy:(fy + height), fx:(fx + width)]
    # print(face.shape)
    # cv2.imshow('face_cut',face)
    # while cv2.waitKey(1) != ord('q'):
    #    pass
    # 缩放图片为32*32
    try:
        face = cv2.resize(face, (cut_row, cut_col))
    except Exception as e:
        print(e)

    else:
        return face, True


# 利用pca 降维  ① data_mate 为样本集合，每一行为一个样本，每一列为样本的一个特征
def zero_mean(data_mate):  # 中心化
    mean_val = np.mean(data_mate, 0)
    new_data = data_mate - mean_val
    return new_data, mean_val
def pca(data_mate, percentage=0.99):
    new_data, mean_val = zero_mean(data_mate)
    # 求中心化后的矩阵的协方差矩阵，rowvar = false 表示 行为样本，列为特征
    cov_mat = np.cov(new_data, rowvar=False)
    # 还有另一种方法求协方差矩阵
    # cov_mat = new_data * new_data.T
    # 求协方差矩阵的特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    # 排列特征值,argsort 是默认升序排列的
    eig_vals_indice = np.argsort(eig_vals)
    # 取出最大的n个特征值，以及算出其对应的特征向量
    n = get_n(eig_vals, percentage)
    n_eig_vals = eig_vals_indice[-1:-(n + 1):-1]
    n_eig_vect = eig_vects[:, n_eig_vals]
    # 特征向量归一化
    # pass
    # 降维
    low_date_mat = new_data * n_eig_vect
    # 重构图像
    # recon_mat = (low_date_mat*n_eig_vect.T) + mean_val
    return low_date_mat, mean_val, n_eig_vect

#pca2 只是另一种方法的测试函数(未完成)不用看
def pca2(data_mate):
    new_data, mean_val = zero_mean(data_mate)
    # 求中心化后的矩阵的协方差矩阵，rowvar = false 表示 行为样本，列为特征
    cov_mat = np.cov(new_data, rowvar=False)
    # 还有另一种方法求协方差矩阵
    # cov_mat = new_data * new_data.T
    # 求协方差矩阵的特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    # 排列特征值,argsort 是默认升序排列的
    eig_vals_indice = np.argsort(eig_vals)
    # 取出最大的n个特征值，以及算出其对应的特征向量
    n = 40
    # n = 30
    n_eig_vals = eig_vals_indice[-1:-(n + 1):-1]
    n_eig_vect = eig_vects[:, n_eig_vals]
    # 特征向量归一化
    # pass
    # 降维
    # low_date_mat = new_data * n_eig_vect
    # 重构图像
    # recon_mat = (low_date_mat*n_eig_vect.T) + mean_val
    return mean_val, n_eig_vect


# 确定要取前n个
def get_n(eig_vals, percentage):
    sort_array = np.sort(eig_vals)  # 升序排序
    sort_array = sort_array[-1::-1]  # 变为降序
    array_sum = sum(sort_array)
    tmpsum = 0
    num = 0
    for eig_val in sort_array:
        tmpsum += eig_val
        num += 1
        if tmpsum >= array_sum * percentage:
            return num


# 把一张图片转换成 1*n 矩阵，之后再把所有m个样本都合并为 m*n 的矩阵，m表示样本数，n表示样本的维数(特征数)
# images 为之前已经裁剪好的训练组列表，格式固定，大小固定
# 单张图片
def img2vector(image):
    # 转为灰度图，减小运算量
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    image_vector = np.zeros((1, rows * cols))  # create a none vector to raise speed
    image_vector = np.reshape(image, (1, rows * cols))  # change img from 2D to 1D
    return image_vector


def load_train_test(total, n):  # 选择每个人从10张图片中选择n张作为训练样本,总共有total人
    print('数据处理开始……')
    data_dir = 'F:/python_study/KL_face/att_faces'
    choose = [num for num in range(1, 11)]  # 给每个人的十张照片编号
    random.shuffle(choose)  # 打乱编号
    # 训练集、训练集中照片所对应的人的编号
    train_face = np.zeros((total * n, cut_row * cut_col))
    train_face_num = np.zeros(total * n)
    # 测试集、以及编号
    test_face = np.zeros((total * (10 - n), cut_row * cut_col))
    test_face_num = np.zeros(total * (10 - n))
    index_train = 0
    index_test = 0
    # flag = True
    for i in range(total):  # 处理 total个人
        people_num = i + 1  # 给第i个人编号，i从0开始，所以要加1
        for j in range(10):
            # 文件路径： F:/python_study/KL_face/att_faces/s1/1.pgm
            # image = np.zeros((0, 0))
            file_name = data_dir + '/s' + str(people_num) + '/' + str(choose[j]) + '.pgm'
            image = cv2.imread(file_name, 0)
            if j < n:
                # 先裁剪成32*32，然后再转换成1*n向量
                image, flag = img_cut(image)

                if flag:
                    # cv2.imshow('1', image)
                    # print(image1.shape)
                    # cv2.waitKey(10)
                    image = img2vector(image)
                    train_face[index_train, :] = image
                    train_face_num[index_train] = people_num
                    index_train += 1
                else:
                    print('drop this photo from train:', file_name)

            else:
                # 先裁剪成32*32，然后再转换成1*n向量
                image, flag = img_cut(image)

                if flag:
                    # cv2.imshow('1', image)
                    # print(image1.shape)
                    # cv2.waitKey(10)
                    image = img2vector(image)
                    test_face[index_test, :] = image
                    test_face_num[index_test] = people_num
                    index_test += 1
                else:
                    print('drop this photo from test:', file_name)
            '''
            # 保存裁剪后的图片
            if flag:
                image = np.reshape(image, (cut_row,cut_col))
                cv2.imshow(str(j),image)
                cv2.waitKey(200)
                cv2.imwrite('F:/python_study/KL_face/att_faces/%d.jpg' % j,image)
                # image_show = Image.fromarray(image)
                # image_show.show()
            '''
    return train_face, train_face_num, test_face, test_face_num


# 人脸识别函数：改变人脸样本数来观察准确率变化
def face_distinguish(total, n):
    train_face, train_face_num, test_face, test_face_num = load_train_test(total, n)
    # PCA training to train_face
    train_face_new, data_mean, W = pca(train_face)
    # W就是特征脸，输出观察一下
    pc_view(W)
    # 得到训练集、测试集的总数
    num_train = train_face_new.shape[0]
    num_test = test_face.shape[0]
    # 测试集中心化
    temp_face = test_face - np.tile(data_mean, (num_test, 1))
    # 得到测试集降维后的数据
    test_face_new = temp_face * W
    test_face_new = np.array(test_face_new)
    train_face_new = np.array(train_face_new)
    # 识别正确的数目
    true_num = 0
    for i in range(num_test):
        test = test_face_new[i, :]
        distance = train_face_new - np.tile(test, (num_train, 1))
        distance = distance ** 2
        # 把矩阵的每一行相加
        distance = distance.sum(axis=1)
        distance = distance.argsort()
        index_min = distance[0]
        if train_face_num[index_min] == test_face_num[i]:
            true_num += 1
    accuracy = float(true_num) / num_test
    print("训练集大小：", len(train_face_num))
    print("测试集大小：", len(test_face_num))
    return accuracy


# 显示特征脸
def pc_view(W):
    # 取前20个特征向量
    row, col = W.shape
    if col != 1:
        col = col
    for i in range(0, 20):
        # 图像归一化
        img = W[:, i]
        img_real = np.array(np.real(img))
        img = np.reshape(img_real, (cut_row, cut_col))
        img = np.array(img)
        img = (img * 255) ** 2
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((128, 128))
        global save_f
        img.save(r'F:\python_study\KL_face\feature\%d.jpg' % save_f)
        save_f += 1
        # img.show()

# 该函数是方法2的人脸识别函数，根据改变最终降维的维度来算准确率(未完成)
def face_distinguish2(train_face, train_face_num, test_face, test_face_num, data_mean, W, n):
    # PCA training to train_face
    # W就是特征脸，输出观察一下
    W = W[:, 0:n]
    # pc_view(W)
    # 得到训练集、测试集的总数
    num_train = train_face.shape[0]
    num_test = test_face.shape[0]
    # 测试集中心化
    temp_face = test_face - np.tile(data_mean, (num_test, 1))
    # 得到测试集降维后的数据
    test_face_new = temp_face * W
    test_face_new = np.array(test_face_new)
    # 得到训练集降维后的数据
    train_face_new = train_face * W
    train_face_new = np.array(train_face_new)
    # 识别正确的数目
    true_num = 0
    for i in range(num_test):
        test = test_face_new[i, :]
        distance = train_face_new - np.tile(test, (num_train, 1))
        distance = distance ** 2
        distance = distance.sum(axis=1)
        distance = distance.argsort()
        index_min = distance[0]
        if train_face_num[index_min] == test_face_num[i]:
            true_num += 1
    accuracy = float(true_num) / num_test
    print("训练集大小：", len(train_face_num))
    print("测试集大小：", len(test_face_num))
    return accuracy


if __name__ == '__main__':

    test_result = {}
    # 计算样本数20到40的准确率
    for test_n in range(20, 45,5):
        a_accuracy = face_distinguish(test_n, 5)
        test_result[test_n] = '%.2f' %a_accuracy
        print('准确率为：', a_accuracy)
    print('最终结果为：')

    test_result=sorted(test_result.items(), key=lambda e: e[0])
    print(test_result)
    x = [0,5]
    y = [0.00,0.6]
    for key,item in test_result:
        x.append(key)
        y.append(item)
    print(x)
    print(y)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, "b-", linewidth=5)
    plt.xlabel("num of image")
    plt.ylabel("accuracy")
    plt.title("PCA")
    plt.show()


    # 下面是改变降维后的维度来计算准确率(未完成)
    '''
    for weidu in range(5, 45, 5):
        m_train_face, m_train_face_num, m_test_face, m_test_face_num = load_train_test(40, 5)
        m_data_mean, m_W = pca2(m_train_face)
        a_accuracy = face_distinguish2(m_train_face, m_train_face_num, m_test_face, m_test_face_num, m_data_mean, m_W,
                                       weidu)
        print('准确率为：', a_accuracy)
        test_result[weidu] = '%.2f' % a_accuracy

    print('最终结果为：')

    test_result = sorted(test_result.items(), key=lambda e: e[0])
    print(test_result)
    x = [0]
    y = [0.00]
    for key, item in test_result:
        x.append(key)
        y.append(item)
    print(x)
    print(y)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, "b--", linewidth=1)
    plt.xlabel("num of image")
    plt.ylabel("accuracy")
    plt.title("PCA")
    plt.show()
    '''