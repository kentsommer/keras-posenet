import helper
import posenet
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Variables
    batch_size = 75

    # Train model
    model = posenet.create_posenet('posenet.npy', True) # GoogLeNet (Trained on Places)
    adam = Adam(lr=0.001, clipvalue=1.5)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    dataset_train, dataset_test = helper.getKings()

    X_train = np.squeeze(np.array(dataset_train.images))
    y_train = np.squeeze(np.array(dataset_train.poses))

    y_train_x = y_train[:,0:3]
    y_train_q = y_train[:,3:7]

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    y_test_x = y_test[:,0:3]
    y_test_q = y_test[:,3:7]

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath="checkpoint_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

    model.fit(X_train, [y_train_x, y_train_q, y_train_x, y_train_q, y_train_x, y_train_q],
          batch_size=batch_size,
          nb_epoch=800,
          validation_data=(X_test, [y_test_x, y_test_q, y_test_x, y_test_q, y_test_x, y_test_q]),
          callbacks=[checkpointer])

    model.save_weights("custom_trained_weights.h5")