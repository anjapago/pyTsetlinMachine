/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

#include "ConvolutionalTsetlinMachine.h"

struct TsetlinMachine *CreateTsetlinMachine(int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses, int dlri)
{
	/* Set up the Tsetlin Machine structure */

	struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));

	tm->number_of_clauses = number_of_clauses;

	tm->number_of_features = number_of_features;

	tm->number_of_clause_chunks = (number_of_clauses-1)/32 + 1;

	tm->number_of_patches = number_of_patches;

	tm->clause_output = (unsigned int *)malloc(sizeof(unsigned int) * tm->number_of_clause_chunks);

	tm->output_one_patches = (int *)malloc(sizeof(int) * number_of_patches);

	tm->number_of_ta_chunks = number_of_ta_chunks;

	tm->feedback_to_la = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);

	tm->number_of_state_bits = number_of_state_bits;

	tm->ta_state = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses * number_of_ta_chunks * number_of_state_bits);

	tm->T = T;

	tm->s = s;

	tm->s_range = s_range;

	tm->clause_patch = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	tm->feedback_to_clauses = (int *)malloc(sizeof(int) * tm->number_of_clause_chunks);
	
	tm->clause_weights = (unsigned int *)malloc(sizeof(unsigned int) * number_of_clauses);

	if (((number_of_features) % 32) != 0) {
		tm->filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		tm->filter = 0xffffffff;
	}

	tm->boost_true_positive_feedback = boost_true_positive_feedback;

	tm->weighted_clauses = weighted_clauses;

    tm->dlri = dlri;
	
	tm_initialize(tm);

	return tm;
}

void tm_initialize(struct TsetlinMachine *tm)
{
	/* Set up the Tsetlin Machine structure */

	unsigned int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits-1; ++b) {											
				tm->ta_state[pos] = ~0;
				pos++;
			}
			tm->ta_state[pos] = 0;
			pos++;
		}
		tm->clause_weights[j] = 1;
	}
}

void tm_destroy(struct TsetlinMachine *tm)
{
	free(tm->clause_output);
	free(tm->output_one_patches);
	free(tm->feedback_to_la);
	free(tm->ta_state);
	free(tm->feedback_to_clauses);
	free(tm->clause_weights);
}

static inline void tm_initialize_random_streams(struct TsetlinMachine *tm, int clause)
{
	// Initialize all bits to zero	
	memset(tm->feedback_to_la, 0, tm->number_of_ta_chunks*sizeof(unsigned int));

	int n = tm->number_of_features;
	double p = 1.0 / (tm->s + 1.0 * clause * (tm->s_range - tm->s) / tm->number_of_clauses);

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % (tm->number_of_features);
		while (tm->feedback_to_la[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % (tm->number_of_features);
	    }
		tm->feedback_to_la[f / 32] |= 1 << (f % 32);
	}
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_inc(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (carry == 0)
			break;

		carry_next = ta_state[b] & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] |= carry;
		}
	} 	
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void tm_dec(struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &tm->ta_state[clause*tm->number_of_ta_chunks*tm->number_of_state_bits + chunk*tm->number_of_state_bits];

	carry = active;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
		if (carry == 0)
			break;

		carry_next = (~ta_state[b]) & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[b] = ta_state[b] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}

	if (carry > 0) {
		for (int b = 0; b < tm->number_of_state_bits; ++b) {
			ta_state[b] &= ~carry;
		}
	} 
}

/* Sum up the votes for each class */
static inline int sum_up_class_votes(struct TsetlinMachine *tm)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		if (j % 2 == 0) {
			class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		} else {
			class_sum -= tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		}	
	}

	class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;
	class_sum = (class_sum < -(tm->T)) ? -(tm->T) : class_sum;

	return class_sum;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
static inline void tm_calculate_clause_output(struct TsetlinMachine *tm, unsigned int *Xi, int predict)
{
	int output_one_patches_count;

	unsigned int *ta_state = tm->ta_state;
	int dlri = tm->dlri;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
		tm->clause_output[j] = 0;
	}
    //printf("LIB: tm_calculate_clause_output ");
	//printf("number of ta chunks: %d \n", (tm->number_of_ta_chunks-1));
    //printf("number of patches: %d \n", (tm->number_of_patches));
	for (int j = 0; j < tm->number_of_clauses; j++) {
		output_one_patches_count = 0;
		for (int patch = 0; patch < tm->number_of_patches; ++patch) {
			unsigned int output = 1;
			unsigned int outputd = 1;
			unsigned int all_exclude = 1;

			for (int k = 0; k < tm->number_of_ta_chunks-1; k++) {
			    //printf("num ta chunk %d \n", k);
				unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;
				// actions for all the ta in the chunk are: ta_state[pos]
//				printf("output %d, ta_state[pos] %d ", output, ta_state[pos]);
//				printf("(ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) %d \n ", (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]));
//				printf("ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos] %d \n ", ((ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos]));
//              printf("output && (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos] %d \n ", (output && (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos]));

                output = output && (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + k]) == ta_state[pos];
                printf("original output: %d \n ", output);

				if  (dlri){
				    // for each ta in the chunk, check the output of its clause, and track it in output
                    for (int chunk_pos=0; chunk_pos<32; chunk_pos++){
                        // get the action for the ta
                        //int ta_num = 1; // the position of the ta within the chunk: "chunk_pos"
                        int action = (tm->ta_state[pos] & (1 << chunk_pos)) > 0; //tm_ta_action(tm, j, k*32+chunk_pos, 1); // j is the clause number

                        // and it with the Xii: if action is 1, check if Xi feature is there, if its 0, don't need to check
                        int clause_output = 1;

                        if (action){ // if action is to include the feature, check if the feature is present
                            int feature_bits = Xi[patch*tm->number_of_ta_chunks + k];
                            // check the bit at position ta_num
                            int bit = (feature_bits >> chunk_pos) & 1U;

                            if(!bit){
                                clause_output = 0;
                            }
                        }
                        // && it with the current output and save it in output again
                        outputd = outputd && clause_output;
                    }
                    printf("dlir output: %d \n ", outputd);
				}

				if (!outputd) {
					break;
				}
				all_exclude = all_exclude && (ta_state[pos] == 0);
			}

            // do this again for the last ta chunk:
//            unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + (tm->number_of_ta_chunks-1)*tm->number_of_state_bits + tm->number_of_state_bits-1;
//            outputd = outputd &&
//                      (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + tm->number_of_ta_chunks - 1] & tm->filter) ==
//                      (ta_state[pos] & tm->filter);

            unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + (tm->number_of_ta_chunks-1)*tm->number_of_state_bits + tm->number_of_state_bits-1;
            int k = tm->number_of_ta_chunks - 1;
			if(!dlri){
//
//                output = output &&
//                      (ta_state[pos] & Xi[patch*tm->number_of_ta_chunks + tm->number_of_ta_chunks - 1] & tm->filter) ==
//                      (ta_state[pos] & tm->filter);
                for (int chunk_pos=0; chunk_pos<32; chunk_pos++){
                    // get the action for the ta
                    //int ta_num = 1; // the position of the ta within the chunk: "chunk_pos"
                    int action = tm_ta_action(tm, j, k*32+chunk_pos, 1); // j is the clause number

                    // and it with the Xii: if action is 1, check if Xi feature is there, if its 0, don't need to check
                    int clause_output = 1;

                    if (action){
                        int feature_bits = Xi[patch*tm->number_of_ta_chunks + k];
                        // check the bit at position ta_num
                        int bit = (feature_bits >> chunk_pos) & 1U;
                        bit = (tm->filter >> chunk_pos) & 1U;

                        if(!bit){
                            clause_output = 0;
                        }
                    }
                    // && it with the current output and save it in output again
                    output = output && clause_output;
                }

                all_exclude = all_exclude && ((ta_state[pos] & tm->filter) == 0);

                output = output && !(predict == PREDICT && all_exclude == 1);

                if (output) {
                    tm->output_one_patches[output_one_patches_count] = patch;
                    output_one_patches_count++;
                }
            }else{
                for (int chunk_pos=0; chunk_pos<32; chunk_pos++){
                    // get the action for the ta
                    //int ta_num = 1; // the position of the ta within the chunk: "chunk_pos"
                    int action = tm_ta_action(tm, j, k*32+chunk_pos, 1); // j is the clause number

                    // and it with the Xii: if action is 1, check if Xi feature is there, if its 0, don't need to check
                    int clause_output = 1;

                    if (action){
                        int feature_bits = Xi[patch*tm->number_of_ta_chunks + k];
                        // check the bit at position ta_num
                        int bit = (feature_bits >> chunk_pos) & 1U;
                        bit = (tm->filter >> chunk_pos) & 1U;

                        if(!bit){
                            clause_output = 0;
                        }
                    }
                    // && it with the current output and save it in output again
                    outputd = outputd && clause_output;
                }
                //printf("dlir output: %d \n ", outputd);

                all_exclude = all_exclude && ((ta_state[pos] & tm->filter) == 0);

                outputd = outputd && !(predict == PREDICT && all_exclude == 1);

                if (outputd) {
                    tm->output_one_patches[output_one_patches_count] = patch;
                    output_one_patches_count++;
                }

            }

		}
	
		if (output_one_patches_count > 0) {
			unsigned int clause_chunk = j / 32;
			unsigned int clause_chunk_pos = j % 32;

 			tm->clause_output[clause_chunk] |= (1 << clause_chunk_pos);

 			int patch_id = fast_rand() % output_one_patches_count;
	 		tm->clause_patch[j] = tm->output_one_patches[patch_id];
 		}
 	}
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_clauses(struct TsetlinMachine *tm, unsigned int *Xi, int class_sum, int target)
{
	unsigned int *ta_state = tm->ta_state;

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

	 	tm->feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)FAST_RAND_MAX) <= (1.0/(tm->T*2))*(tm->T + (1 - 2*target)*class_sum)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}
		
		if ((2*target-1) * (1 - 2 * (j & 1)) == -1) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
			}
		} else if ((2*target-1) * (1 - 2 * (j & 1)) == 1) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type Ia Feedback

				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
			} else {
				// Type Ib Feedback
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

void tm_update(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
    //printf("LIB: tm_update");
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes(tm);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/
	
	tm_update_clauses(tm, Xi, class_sum, target);
}

int tm_score(struct TsetlinMachine *tm, unsigned int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes(tm);
}

int tm_ta_state(struct TsetlinMachine *tm, int clause, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits;
//    printf("In tm_ta_state: ----------------- \n");
	int state = 0;
	for (int b = 0; b < tm->number_of_state_bits; ++b) {
//        printf("tm->ta_state[pos+b]: %d \n", tm->ta_state[pos + b]);
//        printf("(1 << chunk_pos): %d \n", (1 << chunk_pos));
//        printf("tm->ta_state[pos + b] & (1 << chunk_pos): %d \n", tm->ta_state[pos + b] & (1 << chunk_pos));
		if (tm->ta_state[pos + b] & (1 << chunk_pos)) { // checks for ta at certain position within the chunk
			state |= 1 << b;
//            printf("1 << b: %d \n", (1 << b));
//            printf("state: %d \n", state);
		}
	}

	return state;
}

int tm_ta_action(struct TsetlinMachine *tm, int clause, int ta, int print)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits + tm->number_of_state_bits-1;
//	printf("In tm_ta_action: ----------------- \n");
//	printf("tm->ta_state[pos]: %d \n", tm->ta_state[pos]);
//    printf("(1 << chunk_pos): %d \n", (1 << chunk_pos));
//    printf("(tm->ta_state[pos] & (1 << chunk_pos)): %d \n", (tm->ta_state[pos] & (1 << chunk_pos)));
//    printf("tm_ta_action: %d \n", (tm->ta_state[pos] & (1 << chunk_pos)) > 0);
    int original = (tm->ta_state[pos] & (1 << chunk_pos)) > 0;

    if(tm->dlri){
        int state = tm_ta_state(tm, clause, ta);
        int max_state = pow(2, tm->number_of_state_bits)-1;
        //printf("dlri: in tm_ta_action check state: %d \n", state);

        // use dlri method of choosing action: based on probs of the state:
        float action_prob = (float)state/(float)max_state;


        // choose action based on the action prob:
        float rand_val = ((float)fast_rand())/((float)FAST_RAND_MAX); // a random value between 0 and 1

        int action = rand_val < action_prob;

//        int original_action = (tm->ta_state[pos] & (1 << chunk_pos)) > 0;
//        int action = state > (pow(2, tm->number_of_state_bits-1)-1);
//        if(action != original_action){
//            printf("ACTIONS NOT EQUAL, original: %d, dlri action: %d, state: %d \n", original_action, action, state);
//        }

        if(print){
            if(original==action){
                printf("EQUAL******* ");
            }
            printf("original %d, dlri action: %d ", original, action);
            printf("state: %d, max state: %d, ", state, max_state);
            printf("randval: %f, action_prob: %f \n", rand_val, action_prob);
        }
        return action;
    }

	return (tm->ta_state[pos] & (1 << chunk_pos)) > 0;
}

void tm_get_action(struct TsetlinMachine *tm, unsigned int *ta_action)
{
    for (int clause = 0; clause < tm->number_of_clauses; ++clause) {
        for (int ta_chunk = 0; ta_chunk < tm->number_of_ta_chunks; ++ta_chunk) {
            unsigned int pos = clause * tm->number_of_ta_chunks * tm->number_of_state_bits + ta_chunk * tm->number_of_state_bits + tm->number_of_state_bits-1;
//            for (int b = 0; b < tm->number_of_state_bits; ++b) {
//                ta_state[pos] = tm->ta_state[pos];
//                pos++;
//            }

            ta_action[pos] = (tm->ta_state[pos]); //& (1 << chunk_pos)) > 0;
        }
    }
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void tm_get_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				ta_state[pos] = tm->ta_state[pos];
				pos++;
			}
		}
	}
}

void tm_set_state(struct TsetlinMachine *tm, unsigned int *ta_state)
{
	int pos = 0;
	for (int j = 0; j < tm->number_of_clauses; ++j) {
		for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
			for (int b = 0; b < tm->number_of_state_bits; ++b) {
				tm->ta_state[pos] = ta_state[pos];
				pos++;
			}
		}
	}
}

/**************************************/
/*** The Regression Tsetlin Machine ***/
/**************************************/

/* Sum up the votes for each class */
static inline int sum_up_class_votes_regression(struct TsetlinMachine *tm)
{
	int class_sum = 0;

	for (int j = 0; j < tm->number_of_clauses; j++) {
		int clause_chunk = j / 32;
		int clause_pos = j % 32;

		class_sum += tm->clause_weights[j] * ((tm->clause_output[clause_chunk] & (1 << clause_pos)) > 0);
		
	}
	class_sum = (class_sum > (tm->T)) ? (tm->T) : class_sum;

	return class_sum;
}

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target)
{
	unsigned int *ta_state = tm->ta_state;

	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, UPDATE);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int class_sum = sum_up_class_votes_regression(tm);

	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/
	
	// Calculate feedback to clauses

	int prediction_error = class_sum - target; 

	for (int j = 0; j < tm->number_of_clause_chunks; j++) {
	 	tm->feedback_to_clauses[j] = 0;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

	 	tm->feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)FAST_RAND_MAX) <= pow(1.0*prediction_error/tm->T, 2)) << clause_chunk_pos;
	}

	for (int j = 0; j < tm->number_of_clauses; j++) {
		unsigned int clause_chunk = j / 32;
		unsigned int clause_chunk_pos = j % 32;

		if (!(tm->feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos))) {
			continue;
		}
		
		if (prediction_error > 0) {
			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type II Feedback
				
				if (tm->weighted_clauses && tm->clause_weights[j] > 1) {
					tm->clause_weights[j]--;
				}

				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					unsigned int pos = j*tm->number_of_ta_chunks*tm->number_of_state_bits + k*tm->number_of_state_bits + tm->number_of_state_bits-1;

					tm_inc(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & (~ta_state[pos]));
				}
			}
		} else if (prediction_error < 0) {
			// Type I Feedback

			tm_initialize_random_streams(tm, j);

			if ((tm->clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				// Type Ia Feedback
				
				if (tm->weighted_clauses) {
					tm->clause_weights[j]++;
				}
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					int patch = tm->clause_patch[j];
					if (tm->boost_true_positive_feedback == 1) {
		 				tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k]);
					} else {
						tm_inc(tm, j, k, Xi[patch*tm->number_of_ta_chunks + k] & (~tm->feedback_to_la[k]));
					}
		 			
		 			tm_dec(tm, j, k, (~Xi[patch*tm->number_of_ta_chunks + k]) & tm->feedback_to_la[k]);
				}
			} else {
				// Type Ib Feedback
				
				for (int k = 0; k < tm->number_of_ta_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

void tm_fit_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			tm_update_regression(tm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

int tm_score_regression(struct TsetlinMachine *tm, unsigned int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	tm_calculate_clause_output(tm, Xi, PREDICT);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes_regression(tm);
}

/******************************/
/*** Predict y for inputs X ***/
/******************************/

void tm_predict_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples)
{
	unsigned int step_size = tm->number_of_patches * tm->number_of_ta_chunks;

	unsigned int pos = 0;
	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		y[l] = tm_score_regression(tm, &X[pos]);
		
		pos += step_size;
	}
	
	return;
}
