## Running TCAV

# Install required libraries
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils
import pickle

# Added to plot in gbar
import matplotlib
matplotlib.use('Agg')

import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os 
import tensorflow as tf

# Step 1: Store concept and target class images to local folders

# This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
model_to_run = 'GoogleNet'  
user = 'mariafogh'
# the name of the parent directory that results are stored (only if you want to cache)
project_name = 'tcav_class_excludesmallfiles'
working_dir = "/work3/s174498/SaveCavsActivations/" + user + '/' + project_name
# where activations are stored (only if your act_gen_wrapper does so)
activation_dir =  working_dir+ '/activations/'
# where CAVs are stored. 
# You can say None if you don't wish to store any.
cav_dir = working_dir + '/cavs/'
# where the images live.

source_dir = '/work3/s174498/ImageNet_Data_excludesmallfiles'
bottlenecks = ['mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']  # @param 
# bottlenecks = ['mixed4c', 'mixed4d']  # @param 

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

target = 'zebra'  
#concepts = ["dotted","striped","zigzagged"] 
concepts = ["zigzagged","striped","dotted"] 

# Step 2: Model wrapper

# Create TensorFlow session.
sess = utils.create_session()

# GRAPH_PATH is where the trained model is stored.
GRAPH_PATH = source_dir + "/inception5h/tensorflow_inception_graph.pb"
# LABEL_PATH is where the labels are stored. Each line contains one class, and they are ordered with respect to their index in 
# the logit layer. (yes, id_to_label function in the model wrapper reads from this file.)
# For example, imagenet_comp_graph_label_strings.txt looks like:
# dummy                                                                                      
# kit fox
# English setter
# Siberian husky ...

LABEL_PATH = source_dir + "/inception5h/imagenet_comp_graph_label_strings.txt"

mymodel = model.GoogleNetWrapper_public(sess,
                                        GRAPH_PATH,
                                        LABEL_PATH)

# Step 3: Implement a class that returns activations

act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)

# Step 4: Run TCAV
import absl
absl.logging.set_verbosity(0)
start_num_random_exp = 0
num_random_exp = 500
num_random_concepts_to_pick = 10

## only running num_random_exp = 10 to save some time. The paper number are reported for 500 random runs. 
mytcav = tcav.TCAV(sess,
                   target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp,
                   start_num_random_exp = start_num_random_exp,
                   num_random_concepts_to_pick = num_random_concepts_to_pick)

print ('This may take a while... Go get coffee!')
results = mytcav.run(run_parallel=False, overwrite=True)

# end_num_random_exp=start_num_random_exp+num_random_exp-1
#with open('results_pickle/4c4d_result_random500_' + str(start_num_random_exp) + '_to_' + str(end_num_random_exp) + '.pkl', 'wb') as handle:
    #pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('results_pickle/results_'+str(project_name)+'_'+ str(target)+ '_' + str(num_random_exp) +'_' + str(num_random_concepts_to_pick) +'.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('done!')

# Visualize results
# utils_plot.plot_results(results, num_random_exp=num_random_exp)