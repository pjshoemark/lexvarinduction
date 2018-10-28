# lexvarinduction

Research code for the system for **inducing lexical variables from code-mixed text**, described in:
> [Inducing a lexicon of sociolinguistic variables from code-mixed text](http://homepages.inf.ed.ac.uk/s0938610/papers/wnut18_lexvar+supplement.pdf). Philippa Shoemark, James Kirby and Sharon Goldwater. In _Workshop on Noisy User-Generated Text at EMNLP 2018_.

This code was developed for the bottom-up discovery of lexical alternations from a single corpus containing code-mixed text, based on a small list of example word pairs. The words in each pair should belong to different linguistic codes, but otherwise be semantically and syntactically equivalent.

![alt text](https://github.com/pjshoemark/lexvarinduction/blob/master/system_sketch.png "System sketch")



Our approach is inspired by the work of [Schmidt](http://bookworm.benschmidt.org/posts/2015-10-30-rejecting-the-gender-binary.html) and [Bolukbasi et al.](https://papers.nips.cc/paper/6227-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings), who sought to identify pairs of words that exhibit gender bias in their distributional statistics, but are otherwise semantically equivalent. Our method differs in the details but uses a similar framework: we start with a small set of seed pairs, and use these to extract a ‘linguistic code’ component of the embedding space, which is then used to find and rank additional, analogous pairs.


## Usage


1. **Train word embeddings.**

	 You can use the script [train_word2vec_model.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/train_word2vec_model.py)...

  
	 *Example command:*  

	 ```
	 python train_word2vec_model.py --input_corpus_path '/path_to_project_folder/data/input/corpus.tsv' --output_directory_path '/path_to_project_folder/data/embeddings/' --dimensions 500 --min_word_count 50 --window_length 10
	 ```  

	 *Inputs:*  
	 - a single code-mixed text corpus, with one sentence (or short document) per line, and no additional fields.

	 *Outputs:*
	 - a trained word2vec model 

	 ...or, use embeddings you made earlier (must be stored in a format compatible with gensim).  


2. **Generate new variant pairs**  

	 The script [word2vec_analogy_generation.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/word2vec_analogy_generation.py) extracts a dialect component from a provided list of seed variant pairs, and generates a ranked top-k list of additional variant pairs.  


	 *Example command:*  

	 ```  
	 python word2vec_analogy_generation.py -e 'path_to_project_folder/data/embeddings/embeddings_skipgram_dim500_window10_mincount50' -s 'path_to_project_folder/data/input/seed_variables.tsv' -o 'path_to_project_folder/data/output/'
	 ```  


	 *Inputs:*  
	 - trained word embedding model  
	 - list of seed variables, formatted as in [this example file](https://github.com/pjshoemark/lexvarinduction/blob/master/data/input/seed_variables.tsv)


	 *Outputs:*  
	 - `.json` file containing the dialect_axis_vector, ordered list of 
generated pairs and their scores, and list of pairs which were discarded
due to containing words which occur in too many pairs  
	 - `.log` file recording when the script completed running, what options
were used, and what seed variable were used  
	 - `.pdf` plot of seed variables projected onto dialect component  

	 Output filenames will begin `generated_analogy_pairs_`, with the filename of the input word embedding model appended, followed by the parameter settings that were used.  



	

4. **Evaluate the generated pairs**

	 The script [evaluate_generated_analogies.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/evaluate_generated_analogies.py) runs a command line interface which presents generated pairs that have not been previously evaluated for the user to accept or reject. It then calculates precision-at-k for the top-k list.  


	 *Example command:*  

	 ```
	 python evaluate_generated_analogies.py  -p '/path_to_project_folder/data/output/generated_analogy_pairs_embeddings_skipgram_dim500_window10_mincount50_k10000_t00_m10_ddiff_of_mean_vecs_rsim.json' -s '/path_to_project_folder/data/input/seed_variables.tsv' -o 'path_to_project_folder/data/results/' -g '/path_to_project_folder/data/working/good_analogies.tsv' -b '/path_to_project_folder/data/working/bad_analogies.tsv' -a '/path_to_project_folder/data/working/grey_area_analogies.tsv' -k 100
	 ```  

	 *Inputs:*
	 - `.json` file containing ranked list of generated pairs.  
	 - list of seed variables  
	 
	 *Working files:*
	 - list of pairs already evaluated as 'good'
	 - list of pairs already evaluated as 'bad'
	 - list of comments on 'grey area' pairs

	 *Outputs:*   
	 - `.txt` file containing the precision-at-k score.  


	 (the 'good' and 'bad' pairs files are updated as new pairs are evaluated. The 'grey areas' file is for recording brief notes about pairs which were difficult to evaluate)  

	 Output filename will be the filename of the input `.json` file, with `_precision_at_<k>.txt` appended.  

## Detailed Usage

[train_word2vec_model.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/train_word2vec_model.py)

<dl>
<dd><table frame="void" rules="none">

<tr><td colspan="2"><b>Input File</b></td></tr>

<tr><td colspan="2">
<kbd>-i <var>STRING</var>, --input_corpus_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to corpus
</td></tr>


<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Output Directory</b></td></tr>

<tr><td colspan="2">
<kbd>-o <var>STRING</var>, --output_directory_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to directory where output and log should be stored
</td></tr>

<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Options</b></td></tr>

<tr><td colspan="2">
<kbd>-d <var>INT</var>, --dimensions <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Dimensionality of the embeddings
</td></tr>

<tr><td colspan="2">
<kbd>-m <var>INT</var>, --min_word_count <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Ignore all words with total frequency lower than this
</td></tr>

<tr><td colspan="2">
<kbd>-w <var>INT</var>, --window_length <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Maximum distance between current and predicted word within a sentence
</td></tr>


<tr><td colspan="2">
<kbd>-t <var>INT</var>, --n_threads <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Use this many worker threads to train the model (= faster training with multicore machines)
</td></tr>

<tr><td colspan="2">
<kbd>-s <var>INT</var>, --algorithm <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Set to 1 for skipgram; 0 for cbow
</td></tr>

</table>
</dd>
</dl>


[word2vec_analogy_generation.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/word2vec_analogy_generation.py)

<dl>
<dd><table frame="void" rules="none">

<tr><td colspan="2"><b>Input Files</b></td></tr>

<tr><td colspan="2">
<kbd>-e <var>STRING</var>, --embeddings_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to trained word2vec model stored on disk.
</td></tr>

<tr><td colspan="2">
<kbd>-s <var>STRING</var>, --seed_variables_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to list of seed variables
</td></tr>

<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Output Directory</b></td></tr>

<tr><td colspan="2">
<kbd>-o <var>STRING</var>, --output_directory_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to directory where output and log should be stored</td></tr>

<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Options</b></td></tr>

<tr><td colspan="2">
<kbd>-d <var>STRING</var>, --dialect_axis_method <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Which method to use to identify 'dialect' axis. Choose from: 
	
 - mean_of_diff_vecs  
 - pca_indiv  
 - pca_offsets  

</td></tr>


<tr><td colspan="2">
<kbd>-r <var>STRING</var>, --ranking_method <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Which method to use to score candidate analogy pairs. Choose from:  
  
- sim  
- rejection_ratio  
  
</td></tr>

<tr><td colspan="2">
<kbd>-t <var>FLOAT</var>, --similarity_threshold <var>FLOAT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>We will only consider pairs of words whose cosine similarity (to one another) is at least <i>t</i>. If set to 0, the value of <i>t</i> will be chosen automatically using the seed variables.
</td></tr>

<tr><td colspan="2">
<kbd>-k <var>INT</var>, --k_to_keep <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> We will keep a ranked list of the top <i>k</i> analogy pairs
</td></tr>


<tr><td colspan="2">
<kbd>-m <var>INT</var>, --max_occurences <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> We will filter out pairs which contain words who appear in more than <i>m</i> of the top-<i>k</i> ranked pairs (because if a word appears in many pairs its probably a hub, and not genuinely semantically related to its neighbours)
</td></tr>

<tr><td colspan="2">
<kbd>-n <var>INT</var>, --sample_n_seeds <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Can optionally use a random sample of <i>n</i> of the seed variant pairs. If set to 0 (which it is by default), then just use all of them. 
Otherwise, set an integer value less than the total number of seed pairs.
</td></tr>


</table>
</dd>
</dl>



[evaluate_generated_analogies.py](https://github.com/pjshoemark/lexvarinduction/blob/master/code/evaluate_generated_analogies.py)

<dl>
<dd><table frame="void" rules="none">

<tr><td colspan="2"><b>Input Files</b></td></tr>

<tr><td colspan="2">
<kbd>-p <var>STRING</var>, --generated_pairs_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to json file containing list of automatically generated pairs to be evaluated
</td></tr>

<tr><td colspan="2">
<kbd>-s <var>STRING</var>, --seed_variables_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to list of seed variables
</td></tr>

<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Output Directory</b></td></tr>

<tr><td colspan="2">
<kbd>-o <var>STRING</var>, --output_directory <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td>Path to directory where output and log should be stored
</td></tr>


<tr><td colspan="2"></td></tr>
<tr><td colspan="2"><b>Working Files</b></td></tr>

<tr><td colspan="2">
<kbd>-g <var>STRING</var>, --good_pairs_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Path to list of pairs already evaluated as good
</td></tr>

<tr><td colspan="2">
<kbd>-b <var>STRING</var>, --bad_pairs_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Path to list of pairs already evaluated as bad
</td></tr>

<tr><td colspan="2">
<kbd>-a <var>STRING</var>, --grey_area_pairs_path <var>STRING</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Path to file in which to record
notes about pairs which were difficult to evaluate
</td></tr>


<tr><td colspan="2"></td></tr>

<tr><td colspan="2"><b>Options</b></td></tr>
<tr><td colspan="2">
<kbd>-k <var>INT</var>, --k <var>INT</var></kbd></td>
</tr>
<tr><td>&nbsp;</td><td> Value of <i>k</i> to use for precision-at-k score i.e. how many pairs to evaluate (default 120). If input .json file contains fewer than <i>k</i> generated pairs, then precision-at-t is calculated instead, where <i>t</i> is the total number of pairs that were generated.
</td></tr>

</table>
</dd>
</dl>






## Requirements

* Python 2.7 (sorry)
* gensim 3.1.0
* numpy
* scikit-learn
