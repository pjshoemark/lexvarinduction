from collections import defaultdict
import ujson as json
import numpy as np
import argparse, time, os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

description = 'Semi-automatically evaluate top-k generated analogy pairs. Return precision-at-k.'

# function to ensure a directory exists before trying to write to it
def ensure_directory_exists(path_to_directory):
	try: 
		os.makedirs(path_to_directory)
	except OSError:
		if not os.path.isdir(path_to_directory):
			raise

def init_good_pairs(seed_variables_path, good_pairs_path):
	variables = defaultdict(lambda:defaultdict(list))
	with open(seed_variables_path) as infile:
		for line in infile:
			(variable, variant, is_scot) = line.strip().split('\t')
			is_scot = int(is_scot)
			if is_scot:
				variables[variable]['scot'].append(variant)
			else:
				variables[variable]['std'].append(variant)

	variant_pairs = []
	for variable in variables:
		for std_variant in variables[variable]['std']:
			if ' ' not in std_variant:
				for scot_variant in variables[variable]['scot']:
					if ' ' not in std_variant:
						variant_pairs.append((std_variant, scot_variant))

	with open(good_pairs_path, 'w') as outfile:
		for pair in variant_pairs:
			outfile.write('\t'.join(pair)+'\n')


def load_seed_pairs(seed_variables_path):
	variables = defaultdict(lambda:defaultdict(list))
	with open(seed_variables_path) as infile:
		for line in infile:
			(variable, variant, is_scot) = line.strip().split('\t')
			is_scot = int(is_scot)
			if is_scot:
				variables[variable]['scot'].append(variant)
			else:
				variables[variable]['std'].append(variant)

	variant_pairs = []
	for variable in variables:
		for std_variant in variables[variable]['std']:
			if ' ' not in std_variant:
				for scot_variant in variables[variable]['scot']:
					if ' ' not in std_variant:
						variant_pairs.append((std_variant, scot_variant))

	return variant_pairs


def load_good_and_bad_pairs(good_pairs_path, bad_pairs_path):

	good_pairs = set()
	with open(good_pairs_path, 'r') as infile:
		for line in infile:
			pair = line.strip().split('\t')
			good_pairs.add((pair[0], pair[1]))

	bad_pairs = set()
	with open(bad_pairs_path, 'r') as infile:
		for line in infile:
			pair = line.strip().split('\t')
			bad_pairs.add((pair[0], pair[1]))

	return (good_pairs, bad_pairs)


def load_generated_pairs(generated_pairs_path):

	with open(generated_pairs_path, 'r') as infile:

		results = json.load(infile)

		generated_pairs = results['generated_pairs']
		print 'Loaded {}.\nThere are {} output pairs.'.format(generated_pairs_path, str(len(generated_pairs)))

	return generated_pairs



def evaluate_top_k_pairs(k, output_directory_path, generated_pairs_path, good_pairs_path, bad_pairs_path, grey_area_pairs_path, seed_variables_path):

	seed_pairs = load_seed_pairs(seed_variables_path)
	(good_pairs, bad_pairs) = load_good_and_bad_pairs(good_pairs_path, bad_pairs_path)
	generated_pairs = load_generated_pairs(generated_pairs_path)


	p_values = []
	r_values = []
	k_values = []

	average_precision = 0
	
	n_correct = 0
	n_seen = 0

	for pair in generated_pairs:
		if n_seen == k:
			break

		(w1,w2,sim) = pair
		if (w1,w2) in seed_pairs:
			continue

		n_seen += 1

		w1 = w1.encode('utf-8')
		w2 = w2.encode('utf-8')

		# if this pair has previously been seen and evaluated as good, increment the counter of good pairs.
		if (w1,w2) in good_pairs:
			n_correct += 1
			precision = n_correct / float(n_seen)
			change_in_recall = 1 / float(k)
			p_values.append(precision)
			r_values.append(n_correct / float(k))
			k_values.append(n_seen)
			average_precision += precision * change_in_recall

		

		# if this pair has previously been seen and evaluated as bad, move on to the next pair.
		elif (w1,w2) in bad_pairs:
			p_values.append(n_correct / float(n_seen))
			r_values.append(n_correct / float(k))
			k_values.append(n_seen)



		# if this pair has not previously been seen...
		else:

			# print it to console and request an evaluation.
			response = 3
			while response not in [0,1,2]:
				print '\t'.join([w1,w2])
				response = int(raw_input('Is '+'\t'.join([w1,w2])+' a good pair? Enter 1 for Yes; 0 for No; 2 for Grey Area.\n'))
				if response not in [0,1,2]:
					print "That wasn't a 1, 0, or 2 :("


			# if evaluator indicated pair was a grey area:
			if response == 2:

				# record it in the grey areas file, with an explanatory comment.
				comment = raw_input("Ok, we'll record " + '\t'.join([w1,w2]) +" as a grey area. Enter an explanatory comment:\n")

				# then force the evaluator to evaluate it anyway.
				while response not in [0,1]:
					response = int(raw_input('For now, shall we classify  ' + '\t'.join([w1,w2]) +'  as good or bad? Enter 1 for Good; 0 for Bad.\n'))
					if response not in [0,1]:
						print "That wasn't a 1 or 0 :("

				# append pair, comment, and ultimate evaluation to the file of grey areas.
				with open(options.grey_area_pairs_path, 'a') as grey_outfile:
					grey_outfile.write('\t'.join([w1,w2,comment,str(response)])+'\n')



			# once a new pair has been evaluated:
			if response:
				n_correct += 1
				precision = n_correct / float(n_seen)
				change_in_recall = 1 / float(k)
				p_values.append(precision)
				r_values.append(n_correct / float(k))
				k_values.append(n_seen)
				average_precision += precision * change_in_recall

				# add pair to the set of good analogy pairs:
				good_pairs.add((w1,w2))

				# append pair to the file of good analogy pairs:
				with open(options.good_pairs_path, 'a') as good_outfile:
					good_outfile.write('\t'.join([w1,w2])+'\n')
			else:

				p_values.append(n_correct / float(n_seen))
				r_values.append(n_correct / float(k))
				k_values.append(n_seen)
				# add pair to the set of bad analogy pairs:
				bad_pairs.add((w1,w2))

				# append pair to the file of bad analogy pairs:
				with open(options.bad_pairs_path, 'a') as bad_outfile:
					bad_outfile.write('\t'.join([w1,w2])+'\n')
	
	# at end of above for loop, we should have seen k pairs.
	if n_seen != k:
		print 'WARNING: number of pairs seen not equal to K. Seen ' + str(n_seen) + '; K = '+str(options.k)
		
		options.k = n_seen

	# if we have seen k pairs, calculate the precision-at-k score:

	precision_at_k = n_correct / float(n_seen)


	print '\n----------------------------'
	print 'precision at K = {}: {}'.format(str(options.k), str(precision_at_k))
	print '----------------------------'

	print '\n----------------------------'
	print 'average precision at K = {}: {}'.format(str(options.k), str(average_precision))
	print '----------------------------\n\n'


	
	return (precision_at_k, average_precision, p_values, r_values, k_values)


def write_output_and_log(options, precision_at_k, average_precision):

	# set the path for the output
	ensure_directory_exists(options.output_directory_path+'no_seeds/')
	input_filename = os.path.basename(options.generated_pairs_path).split('.')[0]
	output_path = options.output_directory_path+'no_seeds/'+'{}_precision_and_AP_at_{}'.format(input_filename, options.k) 

	# write the output
	with open(output_path+'.eval', 'w') as outfile:
		outfile.write(str(precision_at_k)+'\n')
		outfile.write(str(average_precision))

	# write the log file
	with open(output_path+'.log', 'w') as logfile:
		logfile.write('Created: '+str(time.ctime())+'\n\n')
		logfile.write('Script: '+os.path.abspath(__file__)+'\n\n')
		logfile.write('Description: '+description+'\n\n')
		logfile.write('Input Generated Pairs: '+options.generated_pairs_path+'\n\n')
		logfile.write('Output Evaluation Score: '+output_path+'.eval\n\n')
		logfile.write('Options:-\n')
		for (option, value) in vars(options).iteritems():
			if option in ['generated_pairs_path', 'output_directory_path']:
				continue
			logfile.write('\t'+option + '\t' + str(value) +'\n')


def plot_precision_recall_curve(p_values,r_values, options):
	fig,ax = plt.subplots(figsize=(25, 25), dpi=300,)

	ax.plot(r_values,p_values)

	# set the path for the output
	ensure_directory_exists(options.output_directory_path+'no_seeds/')
	input_filename = os.path.basename(options.generated_pairs_path).split('.')[0]

	# output filename records what model I used 
	# (e.g. skipgram_dim500_window10_mincount100), as well as the options of 
	# this script. (k, t, m)
	output_path = options.output_directory_path+'no_seeds/'+'{}_precision_recall_curve_up_to_k_{}'.format(input_filename, options.k) 
	plt.savefig(output_path+'.pdf') 
	plt.clf()


def plot_precision_at_k_curve(p_values,k_values, options):
	fig,ax = plt.subplots(figsize=(25, 25), dpi=300,)

	ax.plot(k_values,p_values)

	# set the path for the output
	ensure_directory_exists(options.output_directory_path+'no_seeds/')
	input_filename = os.path.basename(options.generated_pairs_path).split('.')[0]

	# output filename records what model I used 
	# (e.g. skipgram_dim500_window10_mincount100), as well as the options of 
	# this script. (k, t, m)
	output_path = options.output_directory_path+'no_seeds/'+'{}_precision_at_k_curve_up_to_k_{}'.format(input_filename, options.k) 
	plt.savefig(output_path+'.pdf') 
	plt.clf()



if __name__ == "__main__": 


	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--seed_variables_path", type=str, default='/data/input/seed_variables.tsv', help="path to list of seed variables")
	parser.add_argument("-p", "--generated_pairs_path", type=str, default='/data/output/generated_analogy_pairs_embeddings_skipgram_dim500_window10_mincount50_k10000_t00_m10_ddiff_of_mean_vecs_rsim.json', help="path to list of automatically generated variant pairs")
	parser.add_argument("-g", "--good_pairs_path", type=str, default='/data/working/good_analogies.tsv', help="path to list of pairs already evaluated as good")
	parser.add_argument("-b", "--bad_pairs_path", type=str, default='/data/working/bad_analogies.tsv', help="path to list of pairs already evaluated as bad")
	parser.add_argument("-a", "--grey_area_pairs_path", type=str, default='/data/working/grey_area_analogies.tsv', help="path to list of grey area pairs")
	parser.add_argument("-o", "--output_directory_path", type=str, default='/data/results/', help="path to directory where output and log should be stored")
	parser.add_argument("-k", "--k", type=str, default=100, help="value of k to use for precision-at-k score. i.e. how many pairs to evaluate.")
	options = parser.parse_args()

	# comment this out after first run:
	# init_good_pairs(options.seed_variables_path, options.good_pairs_path)
	# actually, don't make this something to comment out, make it an option to set (with default value 'off')

	(precision_at_k, average_precision, p_values, r_values, k_values) = evaluate_top_k_pairs(options.k, options.output_directory_path, options.generated_pairs_path, options.good_pairs_path, options.bad_pairs_path, options.grey_area_pairs_path, options.seed_variables_path)
	write_output_and_log(options, precision_at_k, average_precision)
	plot_precision_recall_curve(p_values,r_values,options)
	plot_precision_at_k_curve(p_values,k_values,options)