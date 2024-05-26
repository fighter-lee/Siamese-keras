import datetime
import os

import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.optimizers import SGD, Adam

from nets.siamese import siamese
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ParallelModelCheckpoint)
from utils.dataloader import Datasets
from utils.utils import get_lr_scheduler, load_dataset, show_config

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":

    train_gpu = [0, 1]  # 使用多个GPU

    dataset_path = "datasets"

    input_shape = [105, 105]

    train_own_data = False

    model_path = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    Init_Epoch = 0
    Epoch = 100
    batch_size = 32

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.9

    lr_decay_type = 'cos'

    save_period = 10

    save_dir = 'logs'

    num_workers = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:{}".format(i) for i in train_gpu])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model_body = siamese(input_shape=[input_shape[0], input_shape[1], 3])
        if model_path != '':
            model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

        model = model_body

        train_ratio = 0.9
        train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, train_own_data, train_ratio)
        num_train = len(train_lines)
        num_val = len(val_lines)

        show_config(
            model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, batch_size, Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': Adam(learning_rate=Init_lr_fit, beta_1=momentum),
            'sgd': SGD(learning_rate=Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        train_dataset = Datasets(input_shape, train_lines, train_labels, batch_size, True)
        val_dataset = Datasets(input_shape, val_lines, val_labels, batch_size, False)

        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        logging = TensorBoard(log_dir=log_dir)
        loss_history = LossHistory(log_dir)
        checkpoint = ModelCheckpoint(
            os.path.join(save_dir, "ep{epoch:03d}-loss.h5"),
            monitor='val_loss', save_weights_only=True, save_best_only=False, save_freq=save_period)
        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=False,
                                          save_freq='epoch')
        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"),
                                          monitor='val_loss', save_weights_only=True, save_best_only=True, save_freq='epoch')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
        callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x=train_dataset,
            steps_per_epoch=epoch_step,
            validation_data=val_dataset,
            validation_steps=epoch_step_val,
            epochs=Epoch,
            initial_epoch=Init_Epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=callbacks
        )
