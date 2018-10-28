import gensim, argparse, time, os
import numpy as np
import ujson as json
from bisect import bisect_left 
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#print 'gensim version:', gensim.__version__ # needs to be 3.1.0

description = """Extract dialect component(s) from seed variables and use to 
generate top-k analogy pairs."""



def ensure_directory_exists(path_to_directory):
	"""
	Ensure that a directory exists at the specified path.
	"""

	# Try to create directory at the specified path. If no OS errors arise,
	# then we're good to go.
	try: 
		os.makedirs(path_to_directory)

	# If there's an error, most likely that's because the directory already 
	# exists, so we're good to go.
	except OSError:

		# if the reason for an OSError is *not* that the directory already
		# exists, then something's gone awry. We'd better raise the error!
		if not os.path.isdir(path_to_directory):
			raise


def normalized(a, axis=-1, order=2):
	"""
	Normalize a vector to unit length and return it as a 1D array.
	"""

	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2==0] = 1

	return a / np.expand_dims(l2, axis)



def get_variants(seed_variables_path):
	"""
	Load file containing seed variables into a nested dictionary, and return 
	that along with a list of all the seed word pairs.
	"""

	# old format:

	# load variables into a nested dictionary of form: 
	# dict[variable_name][scottish_or_not] = list_of_variants
	# variables = defaultdict(lambda:defaultdict(list))
	# with open(seed_variables_path) as infile:
	# 	for line in infile:
	# 		(variable, variant, is_scot) = line.strip().split('\t')
	# 		is_scot = int(is_scot)
	# 		if is_scot:
	# 			variables[variable]['scot'].append(variant)
	# 		else:
	# 			variables[variable]['std'].append(variant)

	# # extract list of all variant pairs 
	# variant_pairs = []
	# for variable in variables:
	# 	for std_variant in variables[variable]['std']:
	# 		if ' ' not in std_variant:
	# 			for scot_variant in variables[variable]['scot']:
	# 				if ' ' not in std_variant:
	# 					variant_pairs.append((std_variant, scot_variant))
	


	# new format:

	# load variables into a nested dictionary of form: 
	# dict[variable_name][scottish_or_not] = list_of_variants
	variables = defaultdict(lambda:defaultdict(list))
	with open(seed_variables_path) as infile:
		for line in infile:
			(variant1, variant2) = line.strip().split('/')
			variables[variant1]['scot'].append(variant1)
			variables[variant1]['std'].append(variant2)

	# extract list of all variant pairs 
	variant_pairs = []
	for variable in variables:
		for std_variant in variables[variable]['std']:
			if ' ' not in std_variant:
				for scot_variant in variables[variable]['scot']:
					if ' ' not in std_variant:
						variant_pairs.append((std_variant, scot_variant))
	
	return (variables, variant_pairs)




def load_word_embeddings(embeddings_path):
	"""
	Load a trained word2vec model from disk, and return the word vectors,
	after normalizing them all to unit length.
	"""

	# load a trained word2vec model from disk:
	model = gensim.models.Word2Vec.load(embeddings_path)

	# The word vectors are stored in a KeyedVectors instance in model.wv. 
	# This separates the read-only word vector lookup operations in 
	# KeyedVectors from the training code in Word2Vec.
	# If you're finished training a model (i.e. no more updates, only 
	# querying), then switch to the gensim.models.KeyedVectors instance in wv
	# to trim unneeded model memory = use much less RAM
	word_vectors = model.wv
	del model

	# normalise all the word vectors to unit length: 
	# WARNING: NOT SURE IF THIS WORKS IN GENSIM VERSION 2.3?
	word_vectors.init_sims(replace=True)

	return word_vectors



def get_seed_vectors(seed_variant_pairs, word_vectors):
	"""
	Return a list of seed word vectors, a list of the individiual seed words 
	themselves, and a list indicating which language variety (0 or 1) each
	seed word belongs to.
	"""

	seed_vectors = []
	seed_variants = []
	Y = []
	for pair in seed_variant_pairs:
		(w1, w2) = pair
		if w1 in word_vectors and w2 in word_vectors:
			seed_variants.append(w1)
			seed_vectors.append(word_vectors[w1])
			Y.append(0)
			seed_variants.append(w2)
			seed_vectors.append(word_vectors[w2])	
			Y.append(1)

	print 'number of seed pairs with vectors:', len(seed_vectors) / 2
	return (seed_vectors, seed_variants, Y)




def orient_dialect_vec(transformed_vectors, classes, dialect_vec):
	"""
	I always want the dialect vec to have L1 at the low end and L2 at high end, 
	so that output pairs are always arranged the same way, i.e. (L1,L2) and 
	not (L2,L1). This function figures out which way the dialect vec is
	oriented and flips it if it's the wrong way around.
	"""

	transformed_vectors_L1 = [transformed_vectors[i] for i in range(len(transformed_vectors)) if classes[i] == 0]
	transformed_vectors_L2 = [transformed_vectors[i] for i in range(len(transformed_vectors)) if classes[i] == 1]

	L1_higher = 0
	for i in range(len(transformed_vectors_L1)):
		if transformed_vectors_L1[i][0] > transformed_vectors_L2[i][0]:
			L1_higher += 1

	# If more than half of the L1 seed words got higher values than their L2 
	# equivalents when projected onto dialect_vec, then flip the dialect_vec.
	if L1_higher > len(transformed_vectors_L1)/2:
		dialect_vec = -dialect_vec[0]

	# if L1 seed words tend to get lower values than their L2 equivalents, 
	# then the dialect_vec is already the right way around.
	else:
		dialect_vec = dialect_vec[0]
		
	return dialect_vec





def get_similarity_threshold(seed_variant_pairs, word_embeddings):
	(vectors, variants, classes) = get_seed_vectors(seed_variant_pairs, word_embeddings)

	vectors_L1 = [vectors[i] for i in range(len(vectors)) if classes[i] == 0]
	vectors_L2 = [vectors[i] for i in range(len(vectors)) if classes[i] == 1]

	sims = []

	for i in range(len(vectors_L1)):
		e1 = vectors_L1[i]
		e2 = vectors_L2[i]
		sim = np.dot(e1,e2)
		sims.append(sim)

	similarity_threshold = np.percentile(sims, 25)

	return similarity_threshold




def get_first_pc_indiv_seeds(seed_variant_pairs, word_vectors):
	"""
	Get first principal component of the set of all the individual seed vectors
	"""

	(vectors, variants, Y) = get_seed_vectors(seed_variant_pairs, word_vectors)

	X = np.array(vectors)
	pca = PCA(n_components=2)
	pca.fit(X)
	component = normalized(pca.components_[0]) 
	component2 = normalized(pca.components_[1]) 

	transformed_vectors = pca.transform(X)
	component = orient_dialect_vec(transformed_vectors, Y, component)

	return (component, component2)



def get_first_pc_pairwise_mean_offsets(seed_variant_pairs, word_vectors):
	"""
	Get first principal component of the seed vectors minus their 
	pairwise mean vectors (Bolukbasi et al's method)
	"""

	(vectors, variants, Y) = get_seed_vectors(seed_variant_pairs, word_vectors)

	diff_vectors = []
	for pair in seed_variant_pairs:
		(w1, w2) = pair
		if w1 in word_vectors and w2 in word_vectors:
			mean_vector = normalized(np.mean(np.array([word_vectors[w1], word_vectors[w2]]), axis=0))[0]
			diff_vectors.append(normalized(word_vectors[w1] - mean_vector)[0])
			diff_vectors.append(normalized(word_vectors[w2] - mean_vector)[0])

	X = np.array(diff_vectors)
	pca = PCA(n_components=2)
	pca.fit(X)
	component = normalized(pca.components_[0]) 
	component2 = normalized(pca.components_[1]) 

	transformed_vectors = pca.transform(np.array(vectors))
	component = orient_dialect_vec(transformed_vectors, Y, component)

	return (component, component2)





def get_diff_of_mean_vecs(seed_variant_pairs, word_vectors):
	"""
	Get difference of the mean vectors of the seed pairs 
	"""

	(vectors, variants, Y) = get_seed_vectors(seed_variant_pairs, word_vectors)

	w1_vecs = []
	w2_vecs = []
	for pair in seed_variant_pairs:
		(w1, w2) = pair
		if w1 in word_vectors and w2 in word_vectors:
			w1_vecs.append(word_vectors[w1])
			w2_vecs.append(word_vectors[w2])

	mean_w1_vec = np.mean(w1_vecs, axis=0)
	mean_w2_vec = np.mean(w2_vecs, axis=0)

	diff_of_mean_vecs = normalized(mean_w1_vec - mean_w2_vec)

	transformed_vectors = [] 
	for vec in vectors:
		transformed_vectors.append([np.dot(vec,diff_of_mean_vecs[0])])

	component = orient_dialect_vec(transformed_vectors, Y, diff_of_mean_vecs)
				
	return component



	
def threshold_dot_products(word_embeddings, similarity_threshold):
	"""
	Return a list of tuples of the indices in the word embeddings matrix of all
	pairs of words vectors whose inner product (i.e. cosine similarity, since
	they're all unit length) is greater than or equal to the specified 
	similarity threshold.
	"""

	# get dot products for all pairs of normed vectors (i.e. cosine sims for 
	# all pairs of vectors):
	dps = np.dot(word_embeddings.syn0, word_embeddings.syn0.T)

	# replace 1's on diagonal with 0's.
	np.fill_diagonal(dps, 0)

	# zero out the upper triangle
	dps = np.tril(dps) # or to zero out lower triangle can do dps = np.triu(sims)

	# get the indices of word pairs whose dot product is gt or = to the similarity threshold
	indices = np.where(dps >= similarity_threshold)


	print 'Thresholded dot products.\t'+str(time.ctime())

	return indices





def insert_score(top_scores, top_pairs, abs_score, raw_score, w1, w2, K):


	if raw_score == abs_score:
		w11 = w2
		w22 = w1
	else:
		w11 = w1
		w22 = w2

	# if we have so far stored fewer than K scores values in the list:
	if len(top_scores) < K:

		# get the index at which we should insert this score value in order 
		# to keep the list sorted:
		index = bisect_left(top_scores, abs_score)

		# insert the score value at the specified index in the top_scores list,
		top_scores.insert(index, abs_score)

		# and insert the wordpair tuple at the same index in the top_pairs 
		# list (if cosine score is negative, i.e. raw_score != abs_score, then 
		# flip around the order of the words, so that all L1 variants are on 
		# one side, and all L2 variants on the other.)
		top_pairs.insert(index, (w11,w22))
	

	# else, if list is full but current score value is larger than the 
	# smallest value currently in the list
	elif abs_score > top_scores[0]:


		# delete the smallest value currently in the list
		# and also delete the corresponding word pair
		del top_scores[0]
		del top_pairs[0]

		# then, as above, get the index at which we should insert this score 
		# value in order to keep the list sorted:
		index = bisect_left(top_scores, abs_score)

		# and insert the score value and word pair in their respective lists 
		# at the appropriate index.
		top_scores.insert(index, abs_score)
		top_pairs.insert(index, (w11,w22))

	return (top_scores, top_pairs)




def get_top_K_analogies_sim(word_vectors, indices, dialect_vec, K):
	"""
	Go through all the pairs of word vectors specified in the list of
	index tuples, and rank the pairs according to the cosine similarity of
	their difference vectors to the dialect_vec. (i.e. the ranking method
	we took from Bolukbasi et al.)
	"""

	# we will keep a sorted list of the K highest cosine sim values
	# and also keep another list of the word pairs corresponding to those 
	# cosine sim values
	top_scores = []
	top_pairs = []

	for i in xrange(len(indices[0])):
	#for i in range(1000): #<-- for testing

		# look up the words from their array indices
		w1 = word_vectors.index2word[indices[0][i]]
		w2 = word_vectors.index2word[indices[1][i]]

		# get cosine sim (i.e. dot product, since they're unit length), of diff
		# between the 2 word vectors, and the principal component
		# (the normalized diff and the pca component are 2d arrays, so have to 
		# either convert them both to 1d, or transpose component) 
		score = np.dot(normalized(word_vectors[w1] - word_vectors[w2])[0], dialect_vec)

		# make it abs value:
		abs_score = abs(score)
		
		# insert into sorted list of top K pairs
		(top_scores, top_pairs) = insert_score(top_scores, top_pairs, abs_score, score, w1, w2, K)

		# every millionth word pair, print out the time, so I can check on 
		# progress.
		if i%1000000 == 0:
			print '\t'.join((str(i), str(time.ctime())))+'\n'

	return (top_scores, top_pairs)



def orient_word_pair(e1,e2,w1,w2,dialect_vec):
	""" 
	Get scalar products of both word embedding vectors with the dialect_vec,
	to figure out which one we should consider the 'L1' variant and which one
	the 'L2' variant.
	"""

	if e1.dot(dialect_vec) > e2.dot(dialect_vec):
		return (e2,e1,w1,w2)
	else:
		return (e1,e2,w2,w1)

def get_top_K_analogies_rejection_ratio(word_vectors, indices, dialect_vec, K):
	"""
	Go through all the pairs of word vectors specified in the list of
	index tuples, and rank the pairs according to the ratio of the cosine
	similarity of the two vectors AFTER `rejecting' the dialect_vec, to their
	original cosine similarity. (i.e. the ranking method we took from 
	Schmidt, but *WITHOUT* Schmidt's method of narrowing down the candidate
	pairs.)
	"""

	# we will keep a sorted list of the K highest cosine sim values
	# and also keep another list of the word pairs corresponding to those 
	# cosine sim values
	top_scores = []
	top_pairs = []

	for i in xrange(len(indices[0])):
	#for i in range(1000): #<-- for testing

		# look up the words from their array indices
		w1 = word_vectors.index2word[indices[0][i]]
		w2 = word_vectors.index2word[indices[1][i]]
		e1 = word_vectors[w1]
		e2 = word_vectors[w2]

		(e1,e2,w1,w2) = orient_word_pair(e1,e2,w1,w2,dialect_vec)

		true_sim = np.dot(e1,e2)
		
		e1_dialectless= normalized(e1 - (e1.dot(dialect_vec) * dialect_vec))[0]
		e2_dialectless = normalized(e2 - (e2.dot(dialect_vec) * dialect_vec))[0]

		dialectless_sim = np.dot(e1_dialectless,e2_dialectless)
		score = (dialectless_sim)/(true_sim)

		# insert into sorted list of top K pairs
		(top_scores, top_pairs) = insert_score(top_scores, top_pairs, score, score, w1, w2, K)
				
		# every millionth word pair, print out the time, so I can check on 
		# progress.
		if i%1000000 == 0:
			print '\t'.join((str(i), str(time.ctime())))+'\n'

	return (top_scores, top_pairs)





def filter_top_K_analogies(top_scores, top_pairs, max_occurences):
	"""
	Find all words that appear in (same slot of) more than max_occurences
	pairs, and discard all pairs that contain any such words. Return
	the final, sorted list of kept pairs, and the list of pairs that were 
	discarded.
	"""

	kept_pairs = []
	discarded_pairs = []


	if max_occurences:
	
		# count how many times each word occurs in a pair
		c = Counter()
		for i in xrange(len(top_pairs)): 
			w1 = top_pairs[i][0]
			w2 = top_pairs[i][1]
			c[w1] += 1
			c[w2] += 1

		# iterate through the pairs, in reverse order (highest score is at end of 
		# list, I'd rather have it at beginning).
		for i in xrange(len(top_pairs),0,-1): 
			i -= 1
			w1 = top_pairs[i][0]
			w2 = top_pairs[i][1]
			score = top_scores[i]

			# discard any pairs which contain words occuring in max_occurences 
			# or more pairs
			if c[w1] >= max_occurences or c[w2] >= max_occurences:
				discarded_pairs.append((w1.encode('utf-8'),w2.encode('utf-8'),float(score)))
			else:
				kept_pairs.append((w1.encode('utf-8'),w2.encode('utf-8'),float(score)))

	else:
		for i in xrange(len(top_pairs),0,-1): 
			i -= 1
			w1 = top_pairs[i][0]
			w2 = top_pairs[i][1]
			score = top_scores[i]

			kept_pairs.append((w1.encode('utf-8'),w2.encode('utf-8'),float(score)))


	return (kept_pairs, discarded_pairs)




def generate_analogies(seed_variables_path, word_embeddings, similarity_threshold, sample_n_seeds, k_to_keep, max_occurences, ranking_method, dialect_axis_method):
	"""
	Wrapper function which calls all the relevant functions to get the 
	dialect component(s), and use them to score and rank word pairs. 
	"""
	print '\n\n\nSTART.\t'+str(time.ctime())

	pc2 = 0
	dialect_vec = 0

	(variables, variant_pairs) = get_variants(seed_variables_path)

	print '\nLoaded embeddings and variables.\t'+str(time.ctime())
	print 'Number of variables: ', len(variant_pairs) 
	print variant_pairs[10]

	if sample_n_seeds:
		# sample N variant pairs uniformly at random:
		variant_pairs = [variant_pairs[j] for j in np.random.choice(len(variant_pairs), size=sample_n_seeds, replace=False)]

	if not similarity_threshold:
		similarity_threshold = get_similarity_threshold(variant_pairs, word_embeddings)

	if dialect_axis_method == 'pca_indiv':
		(dialect_vec, pc2) = get_first_pc_indiv_seeds(variant_pairs, word_embeddings) 
		print '\nDone pca (on individual seed vectors).\t'+str(time.ctime())+'\n'

	elif dialect_axis_method == 'pca_offsets':
		(dialect_vec, pc2) = get_first_pc_pairwise_mean_offsets(variant_pairs, word_embeddings) 
		print "\nDone pca (on seed vectors' offsets from pairwise means).\t"+str(time.ctime())+'\n'

	elif dialect_axis_method == 'diff_of_mean_vecs':
		dialect_vec = get_diff_of_mean_vecs(variant_pairs, word_embeddings) 
		print '\nGot diff of mean vecs.\t'+str(time.ctime())+'\n'

	else:
		raise RuntimeError('Invalid dialect_axis_method')




	indices = threshold_dot_products(word_embeddings, similarity_threshold)

	if ranking_method == 'sim':	
		(top_scores, top_pairs) = get_top_K_analogies_sim(word_embeddings, indices, dialect_vec, k_to_keep)
		print 'Got top '+str(k_to_keep)+ ' unfiltered analogy pairs (using sim ranking method).\t'+str(time.ctime())

	elif ranking_method == 'rejection_ratio':
		(top_scores, top_pairs) = get_top_K_analogies_rejection_ratio(word_embeddings, indices, dialect_vec, k_to_keep)
		print 'Got top '+str(k_to_keep)+ ' unfiltered analogy pairs (using rejection_ratio ranking method).\t'+str(time.ctime())

	else:
		raise RuntimeError('Invalid ranking_method')



	(kept_pairs, discarded_pairs) = filter_top_K_analogies(top_scores, top_pairs, max_occurences)
	print 'Filtered. ' + str(len(kept_pairs)) + ' pairs remain.\t'+str(time.ctime()) 



	return (variant_pairs, dialect_vec, pc2, kept_pairs, discarded_pairs)




def write_output_and_log(options, seed_variant_pairs, dialect_vec, kept_pairs, discarded_pairs):
	"""
	Dump the dialect_vec, generated pairs, and discarded pairs in a json file.
	Write a log file recording all the options and the time & date this script 
	was run. 
	"""

	# set the path for the output
	ensure_directory_exists(options.output_directory_path)
	input_filename = os.path.basename(options.embeddings_path).split('.')[0]

	# output filename records what model I used 
	#(e.g. skipgram_dim500_window10_mincount100), as well as the options 
	# of this script. (k, t, m)
	output_path = options.output_directory_path+'generated_analogy_pairs_{}_k{}_t{}_m{}_d{}_r{}'.format(input_filename, options.k_to_keep, ''.join(str(options.similarity_threshold).split('.')), options.max_occurences, options.dialect_axis_method, options.ranking_method)


	# write the output
	with open(output_path+'.json', 'w') as outfile:
		output = {
			'dialect_vec': dialect_vec.tolist(),
			'generated_pairs': kept_pairs,
			'discarded_pairs': discarded_pairs,
		}
		json.dump(output, outfile)

	# write the log file
	with open(output_path+'.log', 'w') as logfile:
		logfile.write('Created: '+str(time.ctime())+'\n\n')
		logfile.write('Script: '+os.path.abspath(__file__)+'\n\n')
		logfile.write('Description: '+description+'\n\n')
		logfile.write('Input Embeddings: '+options.embeddings_path+'\n\n')
		logfile.write('Seed Pairs Used:- \n') 
		logfile.write('\t'.join([' '.join(pair) for pair in seed_variant_pairs])+'\n\n') 
		logfile.write('Output: '+output_path+'.json\n\n')
		logfile.write('Options:-\n')
		for (option, value) in vars(options).iteritems():
			if option in ['embeddings_path', 'seed_variables_path', 'output_directory_path']:
				continue
			logfile.write('\t'+option + '\t' + str(value) +'\n')

	print 'Saved filtered analogy pairs.\t'+str(time.ctime())



def plot_projections_pc(seed_variant_pairs, word_embeddings, dialect_vec, pc2, options):
	"""
	Plot the projections of the seed vectors onto the first two principal
	components.
	"""

	(vectors, variants, classes) = get_seed_vectors(seed_variant_pairs, word_embeddings)

	variants_L1 = [variants[i] for i in range(len(vectors)) if classes[i] == 0]
	variants_L2 = [variants[i] for i in range(len(vectors)) if classes[i] == 1]

	vectors_L1 = [vectors[i] for i in range(len(vectors)) if classes[i] == 0]
	vectors_L2 = [vectors[i] for i in range(len(vectors)) if classes[i] == 1]

	transformed_vectors_L1 = []
	transformed_vectors_L2 = []


	# project L1 seed words onto pcs 1 and 2
	for i in range(len(variants_L1)):
		e = vectors_L1[i]
		transformed_e_pc1 = np.dot(e, dialect_vec)
		transformed_e_pc2 = np.dot(e, pc2[0])
		transformed_vectors_L1.append([transformed_e_pc1, transformed_e_pc2])
		
	# project L2 seed words onto pcs 1 and 2
	for i in range(len(variants_L2)):
		e = vectors_L2[i]
		transformed_e_pc1 = np.dot(e, dialect_vec)
		transformed_e_pc2 = np.dot(e, pc2[0])
		transformed_vectors_L2.append([transformed_e_pc1, transformed_e_pc2])


	fig,ax = plt.subplots(figsize=(25, 25), dpi=300,)

	# plot projected L1 seed words
	ax.scatter([t[0]  for t in transformed_vectors_L1], [t[1]  for t in transformed_vectors_L1])
	for i, txt in enumerate(variants_L1):
		ax.annotate(str(txt), (transformed_vectors_L1[i][0],transformed_vectors_L1[i][1]))

	# plot projected L2 seed words
	ax.scatter([t[0]  for t in transformed_vectors_L2], [t[1]  for t in transformed_vectors_L2], color = 'r')
	for i, txt in enumerate(variants_L2):
		ax.annotate(str(txt), (transformed_vectors_L2[i][0],transformed_vectors_L2[i][1]))


	# set the path for the output
	ensure_directory_exists(options.output_directory_path)
	input_filename = os.path.basename(options.embeddings_path).split('.')[0]

	# output filename records what model I used 
	# (e.g. skipgram_dim500_window10_mincount100), as well as the options of 
	# this script. (k, t, m)
	output_path = options.output_directory_path+'generated_analogy_pairs_{}_k{}_t{}_m{}_d{}_r{}'.format(input_filename, options.k_to_keep, ''.join(str(options.similarity_threshold).split('.')), options.max_occurences, options.dialect_axis_method, options.ranking_method)
	plt.savefig(output_path+'_pca_plot.pdf') 
	plt.clf()

	print 'Plotted pca projections.\t'+str(time.ctime())









def plot_projections_mean(seed_variant_pairs, word_embeddings, dialect_vec, options):
	"""
	Plot the projections of the seed vectors onto the mean_of_diff_vecs or
	diff_of_means_vec. Y axis is meaningless, just used to space them out
	vertically so can see how the two variants of each pair relate to each
	other horizontally.
	"""

	(vectors, variants, classes) = get_seed_vectors(seed_variant_pairs, word_embeddings)

	variants_L1 = [variants[i] for i in range(len(vectors)) if classes[i] == 0]
	variants_L2 = [variants[i] for i in range(len(vectors)) if classes[i] == 1]

	vectors_L1 = [vectors[i] for i in range(len(vectors)) if classes[i] == 0]
	vectors_L2 = [vectors[i] for i in range(len(vectors)) if classes[i] == 1]

	transformed_vectors_L1 = []
	transformed_vectors_L2 = []


	# project L1 seed words onto dialect vec
	for i in range(len(variants_L1)):
		e = vectors_L1[i]
		transformed_e_mean = np.dot(e, dialect_vec)
		transformed_e_y_coord = i
		transformed_vectors_L1.append([transformed_e_mean, transformed_e_y_coord])
		
	# project L2 seed words onto dialect vec
	for i in range(len(variants_L2)):
		e = vectors_L2[i]
		transformed_e_mean = np.dot(e, dialect_vec)
		transformed_e_y_coord = i + 0.25
		transformed_vectors_L2.append([transformed_e_mean, transformed_e_y_coord])


	fig,ax = plt.subplots(figsize=(25, 25), dpi=300,)

	# plot projected L1 seed words
	ax.scatter([t[0]  for t in transformed_vectors_L1], [t[1]  for t in transformed_vectors_L1])
	for i, txt in enumerate(variants_L1):
		ax.annotate(str(txt), (transformed_vectors_L1[i][0],transformed_vectors_L1[i][1]))

	# plot projected L2 seed words 
	ax.scatter([t[0]  for t in transformed_vectors_L2], [t[1]  for t in transformed_vectors_L2], color = 'r')
	for i, txt in enumerate(variants_L2):
		ax.annotate(str(txt), (transformed_vectors_L2[i][0],transformed_vectors_L2[i][1]))


	# set the path for the output
	ensure_directory_exists(options.output_directory_path)
	input_filename = os.path.basename(options.embeddings_path).split('.')[0]

	# output filename records what model I used 
	# (e.g. skipgram_dim500_window10_mincount100), 
	# as well as the options of this script. (k, t, m)
	output_path = options.output_directory_path+'generated_analogy_pairs_{}_k{}_t{}_m{}_d{}_r{}'.format(input_filename, options.k_to_keep, ''.join(str(options.similarity_threshold).split('.')), options.max_occurences,'', options.dialect_axis_method, options.ranking_method)
	plt.savefig(output_path+'_mean_plot.pdf') 
	plt.clf()


	print 'Plotted mean projections.\t'+str(time.ctime())


if __name__ == "__main__": 

	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--embeddings_path", type=str, default='/data/embeddings/embeddings_skipgram_dim500_window10_mincount50', help="path to trained word2vec model stored on disk")
	parser.add_argument("-s", "--seed_variables_path", type=str, default='/data/input/seed_variables.tsv', help="path to list of seed variables")
	parser.add_argument("-o", "--output_directory_path", type=str, default='/data/output/', help="path to directory where output and log should be stored")
	parser.add_argument("-k", "--k_to_keep", type=int, default=10000, help="we will keep a ranked list of the top k analogy pairs")
	parser.add_argument("-t", "--similarity_threshold", type=float, default=0, help="we will only consider pairs of words whose cosine similarity (to one another) is at least t. If set to 0, threshold is set automatically using seed variables")
	parser.add_argument("-m", "--max_occurences", type=int, default=10, help="we will filter out pairs which contain words who appear in more than m of the top-k ranked pairs (because if a word appears in many pairs its probably a hub, and not genuinely semantically related to its neighbours). No filtering if m set to 0.") 
	parser.add_argument("-n", "--sample_n_seeds", type=int, default=0, help="can optionally use a random sample of n of the seed variant pairs. If set to 0, just use all of them.") 
	parser.add_argument("-r", "--ranking_method", type=str, default='sim', help="what method to use to score candidate analogy pairs. Choose from 'sim' or 'rejection_ratio'") 
	parser.add_argument("-d", "--dialect_axis_method", type=str, default='diff_of_mean_vecs', help="what method to use to identify 'dialect' axis. Choose from  'diff_of_mean_vecs', pca_indiv', or 'pca_offsets'")

	options = parser.parse_args()


	word_embeddings = load_word_embeddings(options.embeddings_path)
	(variant_pairs, dialect_vec, pc2, kept_pairs, discarded_pairs) = generate_analogies(options.seed_variables_path, word_embeddings, options.similarity_threshold, options.sample_n_seeds, options.k_to_keep, options.max_occurences, options.ranking_method, options.dialect_axis_method)
	

	write_output_and_log(options, variant_pairs, dialect_vec, kept_pairs, discarded_pairs)

	if options.dialect_axis_method in ['pca_indiv', 'pca_offsets']:
		plot_projections_pc(variant_pairs, word_embeddings, dialect_vec, pc2, options)
	else:
		plot_projections_mean(variant_pairs, word_embeddings, dialect_vec, options)
