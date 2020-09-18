import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import pandas as pd
from som_methods import SelfOrganizingMap
from sklearn.preprocessing import StandardScaler
import logging
from scipy.spatial import distance_matrix
from functions import eucdist

wd = os.getcwd()
#df_x = pd.read_csv(wd + '\\Data\\train_10k_x.csv', delimiter=',', header=None, index_col=False)
#df_y = pd.read_csv(wd + '\\Data\\train_10k_y.csv', delimiter=',', header=None, index_col=False)
#df_z = pd.read_csv(wd + '\\Data\\train_10k_z.csv', delimiter=',', header=None, index_col=False)
#test_df_x = pd.read_csv(wd + '\\Data\\test_2k_x.csv', delimiter=',', header=None, index_col=False)
#test_df_y = pd.read_csv(wd + '\\Data\\test_2k_y.csv', delimiter=',', header=None, index_col=False)
#test_df_z = pd.read_csv(wd + '\\Data\\test_2k_z.csv', delimiter=',', header=None, index_col=False)

#df = df.iloc[1:]
#df = df.drop(df.columns[0], axis=1)

#tup_x = tuple(df_x.itertuples(index=False, name=None))
#tup_y = tuple(df_y.itertuples(index=False, name=None))
#tup_z = tuple(df_z.itertuples(index=False, name=None))

"""df_list = df.values.tolist()
df_ndarray = []
for elem in df_list:
    elem = np.asarray(elem)
    df_ndarray.append(elem)

df_tuple = list2tuple(df_ndarray)"""

'''
An example usage of the TensorFlow SOM. Loads a data set, trains a SOM, and displays the u-matrix.
'''


def get_umatrix(input_vects, weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights and bmu indices of all the input data points

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray`
    :return: input_size x 1 bmu indices 'ndarray'
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))
    # Get the map distance between each neuron (i.e. not the weight distance).
    neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

    for i in range(m * n):
        # Get the indices of the units which neighbor i
        neighbor_idxs = neuron_distmat[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
        # Get the weights of those units
        neighbor_weights = weights[neighbor_idxs]
        # Get the average distance between unit i and all of its neighbors
        # Expand dims to broadcast to each of the neighbors
        umatrix[i] = distance_matrix(np.expand_dims(weights[i], 0), neighbor_weights).mean()

    bmu_indices = []
    for vect in input_vects:
        min_index = min([i for i in range(len(list(weights)))],
                        key=lambda x: np.linalg.norm(vect -
                                                     list(weights)[x]))
        bmu_indices.append(neuron_locs[min_index])

    return umatrix, bmu_indices


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    graph = tf.Graph()
    with graph.as_default():
        # Make sure you allow_soft_placement, some ops have to be put on the CPU (e.g. summary operations)
        session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        num_inputs = 26867
        dims = 256
        #clusters = 2
        # Makes toy clusters with pretty clear separation, see the sklearn site for more info
        #blob_data = make_blobs(num_inputs, dims, clusters)
        #blob_data_x = tup_x
        #blob_data_y = tup_y
        #blob_data_z = tup_z

        dfx = pd.read_csv(wd + '\\Data\\48a.csv', delimiter=',', header=None, index_col=False)
        npx = dfx[np.arange(0,256,1).tolist()].values.astype(float)
        # Scale the blob data for easier training. Also index 0 because the output is a (data, label) tuple.
        scaler = StandardScaler()
        npx_scaled = scaler.fit_transform(npx)
        #dfx_fin = pd.DataFrame(dfxx_scaled)
        #input_data = scaler.fit_transform(blob_data)
        #tr_x = dfx_fin.iloc[:10000]
        tr_x = npx_scaled[:26867]
        te_x = npx_scaled[26867:]
        #te_x = dfx_fin[10000:12000]
        #input_data_x = scaler.fit_transform(tr_x[np.arange(0,256,1).tolist()])
        #amp_x = te_x.transform(df_x[np.arange(0,256,1).tolist()]).to_numpy()


        #input_data_y = scaler.fit_transform(df_y[np.arange(0, 256, 1).tolist()])
        #amp_y = test_df_y.tran().to_numpy()

        #input_data_z = scaler.fit_transform(df_z[np.arange(0, 256, 1).tolist()])
        #amp_z = test_df_z.fit().to_numpy()

        batch_size = 64

        # Build the TensorFlow dataset pipeline per the standard tutorial.
        #dataset_x = tf.data.Dataset.from_tensor_slices(input_data_x.astype(np.float32))
        dataset_x = tf.data.Dataset.from_tensor_slices(tr_x.astype(np.float32))
        dataset_x = dataset_x.repeat()
        dataset_x = dataset_x.batch(batch_size)
        iterator_x = dataset_x.make_one_shot_iterator()
        next_element_x = iterator_x.get_next()

        #dataset_y = tf.data.Dataset.from_tensor_slices(input_data_y.astype(np.float32))
        #dataset_y = dataset_y.repeat()
        #dataset_y = dataset_y.batch(batch_size)
        #iterator_y = dataset_y.make_one_shot_iterator()
        #next_element_y = iterator_y.get_next()

        #dataset_z = tf.data.Dataset.from_tensor_slices(input_data_z.astype(np.float32))
        #dataset_z = dataset_z.repeat()
        #dataset_z = dataset_z.batch(batch_size)
        #iterator_z = dataset_z.make_one_shot_iterator()
        #next_element_z = iterator_z.get_next()

        # This is more neurons than you need but it makes the visualization look nicer
        m = 28
        n = 28

        # Build the SOM object and place all of its ops on the graph
        som_x = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=30, gpus=0, session=session, graph=graph,
                                input_tensor=next_element_x, batch_size=batch_size, initial_learning_rate=0.1)

        #som_y = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=30, gpus=0, session=session, graph=graph,
                                  #input_tensor=next_element_y, batch_size=batch_size, initial_learning_rate=0.1)

        #som_z = SelfOrganizingMap(m=m, n=n, dim=dims, max_epochs=30, gpus=0, session=session, graph=graph,
                                  #input_tensor=next_element_z, batch_size=batch_size, initial_learning_rate=0.1)

        init_op = tf.global_variables_initializer()
        session.run([init_op])

        # Note that I don't pass a SummaryWriter because I don't really want to record summaries in this script
        # If you want Tensorboard support just make a new SummaryWriter and pass it to this method
        som_x.train(num_inputs=num_inputs)
        #som_y.train(num_inputs=num_inputs)
        #som_z.train(num_inputs=num_inputs)

        weights_x = som_x.output_weights
        #weights_y = som_y.output_weights
        #weights_z = som_z.output_weights

        umatrix_x, bmu_loc_x = get_umatrix(tr_x, weights_x, m, n)
        fig_x = plt.figure()
        plt.imshow(umatrix_x.reshape((m, n)), origin='lower')
        plt.show(block=True)

        #umatrix_y, bmu_loc_y = get_umatrix(input_data_y, weights_y, m, n)
        #fig_y = plt.figure()
        #plt.imshow(umatrix_y.reshape((m, n)), origin='lower')
        #plt.show(block=True)

        #umatrix_z, bmu_loc_z = get_umatrix(input_data_z, weights_z, m, n)
        #fig_z = plt.figure()
        #plt.imshow(umatrix_z.reshape((m, n)), origin='lower')
        #plt.show(block=True)


    """edlist_x = np.array([])
    quantisation_errors_x = np.array([])
    for elem in te_x:
        temp_edlist_x = np.array([])
        for ele in weights_x:
            ed = eucdist(elem, ele)
            temp_edlist_x = np.append(temp_edlist_x,ed)
        quantisation_errors_x = np.append(quantisation_errors_x,min(temp_edlist_x))
        edlist_x = np.append(edlist_x,temp_edlist_x)"""
    """
 #train quantisation errors
    tr_ed_x = []
    tr_qe_x = []
    for elem in tr_x:
        temp_ed_x = []
        for ele in weights_x:
            ed = eucdist(elem, ele)
            temp_ed_x.append(ed)
        tr_qe_x.append(min(temp_ed_x))
        tr_ed_x.append(temp_ed_x)

    tr_qe_x = np.array(tr_qe_x)
    df2_tr_time['vals'] = tr_qe_x
    plt.hist(tr_qe_x, bins=100)
    plt.show()
    df2_tr_time.plot(y=["vals"], figsize=(15, 4))
    plt.show()


    #test quantisation errors
    te_ed_x = []
    te_qe_x = []
    for elem in te_x:
        temp_ed_x = []
        for ele in weights_x:
            ed = eucdist(elem, ele)
            temp_ed_x.append(ed)
        te_qe_x.append(min(temp_ed_x))
        te_ed_x.append(temp_ed_x)

    te_qe_x = np.array(te_qe_x)
    df2_te_time['vals'] = te_qe_x
    plt.hist(te_qe_x, bins=100)
    plt.show()
    df2_te_time.plot(y=["vals"], figsize=(15, 4))
    plt.show()
    #df2_te_time.plot.scatter(x=df2_te_time.index ,y=["vals"], figsize=(15, 4))
    #plt.show()
"""


    """ quantisation_errors_x = np.array(quantisation_errors_x)
    test_x_timeseries.columns = ['timestamp', 'values']
    test_x_timeseries['vals'] = quantisation_errors_x
    test_x_timeseries['timestamp'] = pd.to_datetime(test_x_timeseries['timestamp'])
    test_x_timeseries = test_x_timeseries.set_index('timestamp')

    test_x_timeseries = pd.read_csv(wd + '\\tensorflow-som\\df2_time.csv', delimiter=',', header=None)#, index_col=False)
    test_x_timeseries = pd.to_datetime(test_x_timeseries.col0)#, format="%Y%m%d%H%M%S")
    test_x_timeseries = pd.to_datetime(test_x_timeseries)"""



    """
   edlist_y = np.array([])
    quantisation_errors_y = np.array([])
    for elem in amp_y:
        temp_edlist_y = np.array([])
        for ele in weights_y:
            ed = eucdist(elem, ele)
            temp_edlist_y = np.append(temp_edlist_y, ed)
        quantisation_errors_y = np.append(quantisation_errors_y, min(temp_edlist_y))
        edlist_y = np.append(edlist_y, temp_edlist_y)

    edlist_z = np.array([])
    quantisation_errors_z = np.array([])
    for elem in amp_z:
        temp_edlist_z = np.array([])
        for ele in weights_z:
            ed = eucdist(elem, ele)
            temp_edlist_z = np.append(temp_edlist_z, ed)
        quantisation_errors_z = np.append(quantisation_errors_z, min(temp_edlist_z))
        edlist_z = np.append(edlist_z, temp_edlist_z)"""









