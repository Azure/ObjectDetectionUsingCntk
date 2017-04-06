from __future__ import division
from __future__ import print_function
from past.utils import old_div

import os, pdb
from os.path import join
from helpers import readTable

from cntk import *
from cntk.device import use_default_device #default #gpu, set_default_device
from cntk.initializer import glorot_uniform
from cntk.io import MinibatchSource, ImageDeserializer, CTFDeserializer, StreamDefs, StreamDef
from cntk.io.transforms import scale
from cntk.layers import placeholder, constant
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter, TensorBoardProgressWriter
from cntk.logging.graph import find_by_name, plot



####################################
# CNTK-python wrapper functions
####################################
def create_mb_source(data_set, img_height, img_width, n_classes, n_rois, data_path, randomize):
    # set paths
    map_file   = join(data_path, data_set + '.txt')
    roi_file   = join(data_path, data_set + '.rois.txt')
    label_file = join(data_path, data_set + '.roilabels.txt')
    if not os.path.exists(map_file) or not os.path.exists(roi_file) or not os.path.exists(label_file):
        raise RuntimeError("File '%s', '%s' or '%s' does not exist. " % (map_file, roi_file, label_file))

    # read images
    nrImages = len(readTable(map_file))
    transforms = [scale(width=img_width, height=img_height, channels=3,
                        scale_mode="pad", pad_value=114, interpolations='linear')]
    image_source = ImageDeserializer(map_file, StreamDefs(features = StreamDef(field='image', transforms=transforms)))

    # read rois and labels
    rois_dim  = 4 * n_rois
    label_dim = n_classes * n_rois
    roi_source = CTFDeserializer(roi_file, StreamDefs(
        rois = StreamDef(field='rois', shape=rois_dim, is_sparse=False)))
    label_source = CTFDeserializer(label_file, StreamDefs(
        roiLabels = StreamDef(field='roiLabels', shape=label_dim, is_sparse=False)))

    # define a composite reader
    mb = MinibatchSource([image_source, roi_source, label_source], epoch_size=sys.maxsize, randomize=randomize)
    return (mb, nrImages)


# Defines the Fast R-CNN network model for detecting objects in images
def frcn_predictor(features, rois, n_classes, base_path):
    # model specific variables for AlexNet
    model_file = base_path + "/../../../resources/cntk/AlexNet.model"
    roi_dim = 6
    feature_node_name = "features"
    last_conv_node_name = "conv5.y"
    pool_node_name = "pool3"
    last_hidden_node_name = "h2_d"

    # Load the pretrained classification net and find nodes
    print("Loading pre-trained model...")
    loaded_model = load_model(model_file)
    print("Loading pre-trained model... DONE.")
    feature_node = find_by_name(loaded_model, feature_node_name)
    conv_node    = find_by_name(loaded_model, last_conv_node_name)
    pool_node    = find_by_name(loaded_model, pool_node_name)
    last_node    = find_by_name(loaded_model, last_hidden_node_name)

    # Clone the conv layers and the fully connected layers of the network
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: placeholder()})
    fc_layers   = combine([last_node.owner]).clone(CloneMethod.clone,  {pool_node: placeholder()})

    # Create the Fast R-CNN model
    feat_norm = features - constant(114)
    conv_out  = conv_layers(feat_norm)
    roi_out   = roipooling(conv_out, rois, (roi_dim, roi_dim))
    fc_out    = fc_layers(roi_out)
    #fc_out.set_name("fc_out")

    # z = Dense(rois[0], num_classes, map_rank=1)(fc_out)  # --> map_rank=1 is not yet supported
    W = parameter(shape=(4096, n_classes), init=glorot_uniform())
    b = parameter(shape=n_classes, init=0)
    z = times(fc_out, W) + b
    return z, fc_out


# Initialize and train a Fast R-CNN model
def init_train_fast_rcnn(image_height, image_width, num_classes, num_rois, mb_size, max_epochs, cntk_lr_per_image, l2_reg_weight,
                         momentum_time_constant, base_path, boSkipTraining = False, debug_output=False, tensorboardLogDir = None):

    #make sure we use GPU for training
    if use_default_device().type() == 0:
        print("WARNING: using CPU for training.")
    else:
        print("Using GPU for training.")

    # Instantiate the Fast R-CNN prediction model
    image_input = input_variable((3, image_height, image_width))
    roi_input   = input_variable((num_rois, 4))
    label_input = input_variable((num_rois, num_classes))
    frcn_output, frcn_penultimateLayer = frcn_predictor(image_input, roi_input, num_classes, base_path)

    if boSkipTraining:
        print("Using pre-trained DNN without refinement")
        return frcn_penultimateLayer

    # Create the minibatch source and define mapping from reader streams to network inputs
    minibatch_source, epoch_size = create_mb_source("train", image_height, image_width, num_classes, num_rois,
                                                    base_path, randomize=True)
    input_map = {
        image_input: minibatch_source.streams.features,
        roi_input: minibatch_source.streams.rois,
        label_input: minibatch_source.streams.roiLabels
    }

    # set loss / error functions
    ce = cross_entropy_with_softmax(frcn_output, label_input, axis=1)
    pe = classification_error(frcn_output, label_input, axis=1)
    if debug_output:
        plot(frcn_output, "graph_frcn.png")

    # set the progress printer(s)
    progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs)]
    if tensorboardLogDir != None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboardLogDir, model=frcn_output)
        progress_writers.append(tensorboard_writer)

    # Set learning parameters and instantiate the trainer object
    lr_per_sample = [f/float(num_rois) for f in cntk_lr_per_image]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(frcn_output, (ce, pe), learner, progress_writers)

    # Get minibatches of images and perform model training
    print("Training Fast R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(frcn_output)
    for epoch in range(max_epochs):
        sample_count = 0

        # loop over minibatches in the epoch
        while sample_count < epoch_size:
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size - sample_count), input_map=input_map)
            if sample_count % 100 == 1:
                print("Training in progress: epoch {} of {}, sample count {} of {}".format(epoch, max_epochs, sample_count, epoch_size))
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
        trainer.summarize_training_progress()

        # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        if tensorboardLogDir != None:
            for parameter in frcn_output.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", np.mean(parameter.value), epoch)
                tensorboard_writer.write_value(parameter.uid + "/std", np.std(parameter.value), epoch)
                tensorboard_writer.write_value(parameter.uid + "/absSum", np.sum(np.abs(parameter.value)), epoch)

        if debug_output:
            frcn_output.save_model("frcn_py_%s.model" % (epoch + 1))
    return frcn_output


def run_fast_rcnn(model, data_set, image_height, image_width, num_classes, num_rois, base_path, outDir):
    # Create the minibatch source and define mapping from reader streams to network inputs
    minibatch_source, num_images = create_mb_source(data_set, image_height, image_width, num_classes, num_rois, base_path, randomize=False)
    input_map = {
        model.arguments[0]: minibatch_source['features'],
        model.arguments[1]: minibatch_source['rois']
    }

    # evaluate test images and write to file
    for imgIndex in range(0, num_images):
        if imgIndex % 100 == 1:
            print("Evaluating images {} of {}".format(imgIndex, num_images))
        data = minibatch_source.next_minibatch(1, input_map=input_map)
        output = model.eval(data)[0][0]
        output = np.array(output, np.float32)

        # write to disk
        if imgIndex % 100 == 1:
            print("Writing DNN output of dimension {} to disk".format(output.shape))
        outPath = outDir + str(imgIndex) + ".dat"
        np.savez_compressed(outPath, output)