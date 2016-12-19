import math
import helper
import posenet
import numpy as np
from keras.optimizers import Adam

if __name__ == "__main__":
    # Test model
    model = posenet.create_posenet()
    model.load_weights('trained_weights.h5')
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=2.0)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    dataset_train, dataset_test = helper.getKings()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    testPredict = model.predict(X_test)

    valsx = testPredict[4]
    valsq = testPredict[5]

    # Get results... :/
    results = np.zeros((len(dataset_test.images),2))
    for i in range(len(dataset_test.images)):

        pose_q= np.asarray(dataset_test.poses[i][3:7])
        pose_x= np.asarray(dataset_test.poses[i][0:3])
        predicted_x = valsx[i]
        predicted_q = valsq[i]

        pose_q = np.squeeze(pose_q)
        pose_x = np.squeeze(pose_x)
        predicted_q = np.squeeze(predicted_q)
        predicted_x = np.squeeze(predicted_x)

        #Compute Individual Sample Error
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-predicted_x)
        results[i,:] = [error_x,theta]
        print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta
    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')