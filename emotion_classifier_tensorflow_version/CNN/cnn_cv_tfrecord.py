import tensorflow as tf
import os
import numpy as np
from CNN import cnn

channel = 1  # 图像通道数
default_height = 48  # 图像宽高
default_width = 48  
batch_size = 256  # 批尺寸，内存小就调小些
test_batch_size = 256  # 测试时的批尺寸，内存小就调小些
shuffle_pool_size = 4000 # ???内存小就调小些
generations= 5000  #总迭代数
save_flag = True   #是否保存模型
retrain = False    #是否要继续之前的训练
data_folder_name = '..\\temp'
data_path_name = 'cv'
pic_path_name = 'pic'
record_name_train = 'fer2013_train.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
save_ckpt_name = 'cnn_emotion_classifier.ckpt'# 训练出的模型
model_log_name = 'model_log.txt' # 日志文件，记录训练的结果
tensorboard_name = 'tensorboard' 
tensorboard_path = os.path.join(data_folder_name, data_path_name, tensorboard_name)
model_log_path = os.path.join(data_folder_name, data_path_name, model_log_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)


# 数据增强
#要进行深度学习有时候没有大量数据，为了获得更多数据，我们仅需要对已有的数据集做微小的调整。比如翻转、平移或旋转。神经网络会认为这些数据是不同的。
def pre_process_img(image):
    image = tf.image.random_flip_left_right(image) # 随机翻转  
    image = tf.image.random_brightness(image, max_delta=32./255) # 随机调整亮度函数
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # 随机调整对比度
    image = tf.random_crop(image, [default_height-np.random.randint(0, 4), default_width-np.random.randint(0, 4), 1]) # 将图片裁剪为指定大小
    image = tf.image.resize_images(image, [default_height, default_width]) # 调整图片的大小
    return image


# tfrecord的数据读入部分
def __parse_function_csv(serial_exmp_):
    features_ = tf.parse_single_example(serial_exmp_,
                                        features={"image/label": tf.FixedLenFeature([], tf.int64), #tf.FixedLenFeature 返回的是一个定长的tensor
                                                  "image/height": tf.FixedLenFeature([], tf.int64),
                                                  "image/width": tf.FixedLenFeature([], tf.int64),
                                                  "image/raw": tf.FixedLenFeature([default_width*default_height*channel]
                                                                                  , tf.int64)}) #解析单个 Example 原型.
    label_ = tf.cast(features_["image/label"], tf.int32) #tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32
    height_ = tf.cast(features_["image/height"], tf.int32)
    width_ = tf.cast(features_["image/width"], tf.int32)
    image_ = tf.cast(features_["image/raw"], tf.int32)
    image_ = tf.reshape(image_, [height_, width_, channel])
    image_ = tf.multiply(tf.cast(image_, tf.float32), 1. / 255) #tf.multiply（）两个矩阵中对应元素各自相乘
    image_ = pre_process_img(image_)
    return image_, label_


# 同上
def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_) #从tfrecord文件创建TFRecordDataset：
    return data_set_.map(__parse_function_csv) #map方法可以接受任意函数以对dataset中的数据进行处理；另外，可使用repeat、shuffle、batch方法对dataset进行重复、混洗、分批


# 评估准确度
def evaluate(logits_, y_):
    return np.mean(np.equal(np.argmax(logits_, axis=1), y_)) # 计算acc


def main(argv):
    with tf.Session() as sess: # 创建一个会话
        summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph) #指定一个文件用来保存图。格式：tf.summary.FileWritter(path,sess.graph)可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
        data_set_train = get_dataset(record_name_train) #获得训练集样本，返回样本的image和label
        data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(batch_size).repeat() #将数据打乱，数值越大，混乱程度越大,按照顺序取出4行数据，最后一次输出可能小于batch,数据集重复了指定次数
        data_set_train_iter = data_set_train.make_one_shot_iterator() #单次迭代器，这个方法会返回一个 Iterator 对象。而调用 iterator 的 get_next() 就可以轻松地取出数据了。
        train_handle = sess.run(data_set_train_iter.string_handle())

       
      # 对测试集样本进行获取
        data_set_test = get_dataset(record_name_test)
        data_set_test = data_set_test.shuffle(shuffle_pool_size).batch(test_batch_size).repeat()
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        
        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types, data_set_train.output_shapes) #生成迭代器
        x_input_bacth, y_target_batch = iterator.get_next()

        cnn_model = cnn.CNN_Model() #调用卷积层
        x_input = cnn_model.x_input 
        y_target = cnn_model.y_target
        logits = tf.nn.softmax(cnn_model.logits) #Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
        loss = cnn_model.loss 
        train_step = cnn_model.train_step
        dropout = cnn_model.dropout # 防止过拟合
        sess.run(tf.global_variables_initializer()) #初始化模型的参数

        if retrain:
            print('retraining')
            ckpt_name = 'cnn_emotion_classifier.ckpt'
            ckpt_path = os.path.join(data_folder_name, data_path_name, ckpt_name)
            saver = tf.train.Saver() #将训练好的模型参数保存起来
            saver.restore(sess, ckpt_path)

        with tf.name_scope('Loss_and_Accuracy'):
            tf.summary.scalar('Loss', loss) # 用来显示标量信息，
        summary_op = tf.summary.merge_all()   #merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了

        print('start training')
        saver = tf.train.Saver(max_to_keep=1)
        max_accuracy = 0
        temp_train_loss = []
        temp_test_loss = []
        temp_train_acc = []
        temp_test_acc = []
        for i in range(generations):
            x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: train_handle})
            train_feed_dict = {x_input: x_batch, y_target: y_batch,
                               dropout: 0.5}
            sess.run(train_step, train_feed_dict)
            if (i + 1) % 100 == 0:
                train_loss, train_logits = sess.run([loss, logits], train_feed_dict)
                train_accuracy = evaluate(train_logits, y_batch)
                print('Generation # {}. Train Loss : {:.3f} . '
                      'Train Acc : {:.3f}'.format(i, train_loss, train_accuracy))
                temp_train_loss.append(train_loss)
                temp_train_acc.append(train_accuracy)
                summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
            if (i + 1) % 400 == 0:
                test_x_batch, test_y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: test_handle})
                test_feed_dict = {x_input: test_x_batch, y_target: test_y_batch,
                                  dropout: 1.0}
                test_loss, test_logits = sess.run([loss, logits], test_feed_dict)
                test_accuracy = evaluate(test_logits, test_y_batch)
                print('Generation # {}. Test Loss : {:.3f} . '
                      'Test Acc : {:.3f}'.format(i, test_loss, test_accuracy))
                temp_test_loss.append(test_loss)
                temp_test_acc.append(test_accuracy)
                if test_accuracy >= max_accuracy and save_flag and i > generations // 2:
                    max_accuracy = test_accuracy
                    saver.save(sess, os.path.join(data_folder_name, data_path_name, save_ckpt_name))
                    print('Generation # {}. --model saved--'.format(i))
        print('Last accuracy : ', max_accuracy)
        with open(model_log_path, 'w') as f:
            f.write('train_loss: ' + str(temp_train_loss))
            f.write('\n\ntest_loss: ' + str(temp_test_loss))
            f.write('\n\ntrain_acc: ' + str(temp_train_acc))
            f.write('\n\ntest_acc: ' + str(temp_test_acc))
        print(' --log saved--')


if __name__ == '__main__':
    tf.app.run()
