import gensim, logging, argparse, time, os

description = 'Train word embeddings using gensim word2vec implementation.'

# set up gensim logging (logs to stdout)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# function to ensure a directory exists before trying to write to it
def ensure_directory_exists(path_to_directory):
	try: 
		os.makedirs(path_to_directory)
	except OSError:
		if not os.path.isdir(path_to_directory):
			raise


def train_embeddings(input_corpus_path):

	# load corpus as an iterable that streams the tweets/sentences directly from disk/network
	input_corpus = gensim.models.word2vec.LineSentence(options.input_corpus_path)


	# Initialize and train the word2vec model
	model = gensim.models.word2vec.Word2Vec(
		sentences=input_corpus,
		sg=options.algorithm, 
		size=options.dimensions,
		window=options.window_length, 
		min_count=options.min_word_count,
		workers=options.n_threads,
	)

	return model



def write_output_and_log(options, model):

	# set the path for the output
	ensure_directory_exists(options.output_directory_path)
	if options.algorithm:
		algorithm = 'skipgram'
	else:
		algorithm = 'cbow'
	output_path = options.output_directory_path+'embeddings_{}_dim{}_window{}_mincount{}'.format(algorithm, options.dimensions, options.window_length, options.min_word_count)

	# save the trained word2vec model
	model.save(output_path)

	# write the log file
	with open(output_path+'.log', 'w') as logfile:
		logfile.write('Created: '+str(time.ctime())+'\n\n')
		logfile.write('Script: '+os.path.abspath(__file__)+'\n\n')
		logfile.write('Description: '+description+'\n\n')
		logfile.write('Input: '+options.input_corpus_path+'\n\n')
		logfile.write('Output: '+output_path+'\n\n')
		logfile.write('Options:-\n')
		for (option, value) in vars(options).iteritems():
			if option in ['input_corpus_path', 'output_directory_path']:
				continue
			logfile.write('\t'+option + '\t' + str(value) +'\n')


if __name__ == "__main__": 

	# parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_corpus_path", type=str, default='/data/input/corpus.tsv', help="path to training corpus. corpus should be one tweet (or sentence) per line.")
	parser.add_argument("-o", "--output_directory_path", type=str, default='/data/embeddings/', help="path to directory where trained model and log should be stored")
	parser.add_argument("-d", "--dimensions", type=int, default=500, help="dimensionality of the embeddings")
	parser.add_argument("-m", "--min_word_count", type=int, default=50, help="ignore all words with total frequency lower than this")
	parser.add_argument("-w", "--window_length", type=int, default=10, help="maximum distance between current and predicted word within a sentence")
	parser.add_argument("-t", "--n_threads", type=int, default=10, help="use these many worker threads to train the model (=faster training with multicore machines)")
	parser.add_argument("-s", "--algorithm", type=int, default=1, help="1 for skipgram, 0 for cbow") # gensim documentation says 1 for cbow; 0 for skipgram, but that's a typo!
	options = parser.parse_args()

	model = train_embeddings(options.input_corpus_path)
	write_output_and_log(options, model)
