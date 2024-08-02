import os
import re
import sys
from datetime import datetime

from utils import db_search_from_index, get_response, retrieve_vectors_by_ids, count_tokens, \
                  print_similarities, simple_prompt, SimilarityStatCount, output_wrapper, get_response_tinyllama

base_dir = os.path.dirname(__file__)

def run_w_features_n_centroid(questions, qa_list, dataset_name,
                              num_retrieved_docs,  # number of vectors
                              st_model, st_model_name, st_cache_folder,
                              index_path, chunks_file, log_dir, print_entire_responce,
                              tokenizer, base_model, base_model_name, llm_configs, max_new_tokens,
                              seed_everything_flag,
                              n_components_pca_list,
                              n_clusters_gmm_list,
                              alpha,
                              percentile,
                              feature_method,
                              excluded_outlier_idxs,
                              min_outlier_freq,
                              ):
    ssc_obj = SimilarityStatCount()

    run_reverse = False
    if run_reverse:
        loop = reversed(list(enumerate(questions)))
    else:
        loop = enumerate(questions)
    test_questions = False
    if test_questions:
        questions = [
            'How did the oil embargo affect the U.S. economy?'
            # 'Which organization proclaimed the oil embargo in 1973?'
            # 'What major event occurred on October 6, 1973?',
            # 'How did the oil embargo affect the U.S.economy?',
            # 'What were some of the uses of the increased income from higher oil prices?',
            # 'Which country immediately announced an oil embargo on the United States?',
            # 'What impact did the crisis have on international relations?',
            # 'Who is the owner of Manchester City FC?',
            # 'What is the benchmark bond yield in India?',
            # 'What is the scale of debt held by IL&FS and its subsidiaries, and what are the broader implications?',
        ]
        qa_list = [
            ('_', 'Syria and Egypt, supported by other Arab nations, launched a surprise attack on Israel on Yom Kippur.'),
            ('_', 'It caused immediate demands to address energy security threats and had inflationary and deflationary impacts.'),
            ('_', 'Some income was used for aid to underdeveloped nations and for arms purchases.'),
            ('_', 'Libya announced an oil embargo on the United States.'),
            # ('_', 'It created a rift within NATO and influenced nations to disassociate from U.S. foreign policy in the Middle East.')
            # ('_', 'Manchester City FC is owned by Sheikh Mansour bin Zayed al Nahyan.'),
            # ('_', 'The benchmark 5.85% bond maturing in 2030 ended yielding 6.01%'),
            # ('_', 'IL&FS and its subsidiaries hold a combined debt of over Rs 91,000 crore, highlighting significant financial challenges and potential systemic risk.'),
        ]
    q_number = 0
    step = 1
    for ii, question in loop:

        if not ii // step * step == ii:
            continue
        print('ii = ' + str(ii))

        # gt - stands for Ground Truth
        if (dataset_name == 'SQUAD2' or dataset_name == 'mteb-trec-covid' or
                dataset_name == 'bloomberg_quint_news' or dataset_name == 'stanford_qa'):
            gt_answer = qa_list[ii][1].strip()
        else:
            gt_answer = None

        # get embeddings
        retrieved_docs, dist_list, query_embedding, ids = \
            db_search_from_index(question, st_model_name, st_cache_folder, index_path,
                                 chunks_file, num_retrieved_docs)
        print('Distances list with IDs before filtering:\n',
              '\n'.join([str(round(d, 3)) + '   ' + str(id) for (d, id) in zip(dist_list, ids)]))
        if num_retrieved_docs != len(retrieved_docs):
            print('K != len(retrieved_docs)')
            sys.exit(1)

        print('Number of docs considered:', num_retrieved_docs)

        context_written = False

        # ======== filtered retrieval =================================================
        retrieved_embeddings = retrieve_vectors_by_ids(index_path, ids)
        N = retrieved_embeddings.shape[1]

        if len(retrieved_embeddings) != num_retrieved_docs:
            print('len(retrieved_embeddings) != K')
            sys.exit(1)

        plot_file_feat = \
            (base_dir + '/plots/plot_feat_' +
             str(ii + 1) + '.pdf')

        filtered_docs, filtered_ids, thresholds, outlier_docs, outlier_ids = \
            output_wrapper(retrieved_docs, retrieved_embeddings, ids, query_embedding, alpha, plot_file_feat,
                           n_components_pca_list=n_components_pca_list,
                           n_clusters_gmm_list=n_clusters_gmm_list,
                           min_outlier_freq = min_outlier_freq,
                           percentile=percentile,
                           feature_method=feature_method,
                           excluded_outlier_idxs=excluded_outlier_idxs
                           )
        print('\nquestion:', question)
        print()

        num_filtered_docs = len(filtered_docs)
        print('Number of filtered docs:', num_filtered_docs)

        num_orig_docs = num_filtered_docs

        if num_filtered_docs < 1:
            print('All docs have been filtered out')
            print('Question:\n ', question)
            print('Ground Truth answer:\n ', gt_answer)
            continue
        # count = ssc_obj.do_count()
        context_filt = '\n'.join(filtered_docs)
        respond_time_filt = None
        if 'TinyLlama' in base_model_name:
            response_filt, num_words_in_prompt_filt, respond_time_filt = get_response_tinyllama(tokenizer, question, context_filt, base_model, max_new_tokens=max_new_tokens)
            if response_filt == 'Not Found':
                print('Answer to question:', '"' + question + '" Not Found')
                continue
        else:
            prompt = simple_prompt(question, context_filt)
            response_filt, num_words_in_prompt_filt = get_response(tokenizer, prompt, base_model, llm_configs)

        if print_entire_responce:
            print('\nFiltered response:\n', response_filt)
        # =============================================================================

        # ======== original retrieval with num_filtered_docs ==========================
        retrieved_docs, dist_list, query_embedding, ids = \
            db_search_from_index(question, st_model_name,
                                 st_cache_folder, index_path,
                                 chunks_file,
                                 num_orig_docs
                                 )
        print('Number of original docs:', num_orig_docs)

        context_orig = '\n'.join(retrieved_docs)
        respond_time_orig = None
        if 'TinyLlama' in base_model_name:
            response_orig, num_words_in_prompt_orig, respond_time_orig = get_response_tinyllama(tokenizer, question, context_orig, base_model, max_new_tokens=max_new_tokens)
            if response_filt == 'Not Found':
                print('Answer to question:', '"' + question + '" Not Found')
                continue
        else:
            prompt = simple_prompt(question, context_orig)
            response_orig, num_words_in_prompt_orig = get_response(tokenizer, prompt, base_model, llm_configs)

        if print_entire_responce:
            print('\nRegular response:\n', response_orig, '\n\n')
        # =============================================================================
        count = ssc_obj.do_count()

        reg = r'^.*?\n<Answer:>\s*'
        answer_orig = re.sub(reg, '', response_orig, flags=re.S)
        answer_filt = re.sub(reg, '', response_filt, flags=re.S)
        num_tokens_answer_orig = count_tokens(answer_orig)
        num_tokens_answer_filt = count_tokens(answer_filt)

        date = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        ssc_obj.count_similarities(answer_orig, answer_filt, gt_answer, st_model=st_model)
        counts_str = ssc_obj.print_counts(mode='string')
        sim_string = print_similarities(answer_orig, answer_filt, gt_answer, st_model)

        average_reduction = round(ssc_obj.count_av_reduction(num_retrieved_docs, num_filtered_docs, count), 3)
        av_sims = ssc_obj.get_av_similarities(answer_orig, answer_filt, gt_answer, st_model)
        av_sims = [round(elem, 3) for elem in av_sims]
        av_num_filtered_docs = round(ssc_obj.get_av_num_filtered_docs(num_filtered_docs), 3)

        av_sim_change_emb, av_sim_change_tfidf = ssc_obj.get_av_sim_change()
        av_sim_change_emb = round(av_sim_change_emb * 100., 3)
        av_sim_change_tfidf = round(av_sim_change_tfidf * 100., 3)

        q_number += 1
        count_str = 'Question number: ' + str(q_number) + '  out of: ' + str(len(questions))
        rec = f'''======= Begin =================================================================
    {date}
{count_str}
Dataset:                            {dataset_name}
seed_everything_flag:               {seed_everything_flag}
LLM model name (base):              {base_model_name}
Embeddings model name:              {st_model_name}

ii:                                 {ii}
Vector size:                        {N}                 
Number of retrieved docs:           {num_retrieved_docs}

Percentile:                         {percentile}

feature_method:                     {feature_method}
excluded_outlier_idxs:              {excluded_outlier_idxs}
alpha:                              {alpha}

respond_time_filt:                  {respond_time_filt}
respond_time_orig:                  {respond_time_orig}

Number of original docs:            {num_orig_docs}
Number of filtered docs:            {num_filtered_docs}
Average number of filtered docs:    {av_num_filtered_docs}

Average reduction:                  {average_reduction}

num_words_in_prompt_orig:           {num_words_in_prompt_orig}
num_words_in_prompt_filt:           {num_words_in_prompt_filt}

n_components_pca_list:              {n_components_pca_list}  
n_clusters_gmm_list:                {n_clusters_gmm_list} 
min_outlier_freq:                   {min_outlier_freq}

Average Similarity to Ground Truth:
Embedding orig:                     {av_sims[0]}
Embedding filt:                     {av_sims[1]}
TFIDF orig:                         {av_sims[2]}
TFIDF filt:                         {av_sims[3]}

Average Similarity Changes:
Embedding:                          {av_sim_change_emb}%
TFIDF:                              {av_sim_change_tfidf}%

max_new_tokens:                     {max_new_tokens}

num_tokens_answer_orig:             {num_tokens_answer_orig}
num_tokens_answer_filt:             {num_tokens_answer_filt}

Question:\n {question}

Ground Truth answer:\n  {gt_answer}
Original answer:\n  {answer_orig}
Filtered answer:\n  {answer_filt}

Outlier docs:\n     {outlier_docs}
Outlier ids:\n      {outlier_ids}

Similarities to Ground Truth current:
{sim_string}

counts: 
{counts_str}
======= End ===================================================================
'''
        lf = open(log_dir + '/log.txt', 'a', encoding='utf-8')
        lf_cont = open(log_dir + '/log_context.txt', 'a', encoding='utf-8')

        print(rec)
        lf.write(rec)
        lf.close()

        if not context_written:  # write context only once
            lf_cont.write('========================================')
            lf_cont.write('\n')
            lf_cont.write(date)
            lf_cont.write('\n\n')
            lf_cont.write('Question:\n' + question)
            lf_cont.write('\n\n')
            lf_cont.write('Context filt:\n' + context_filt)
            lf_cont.write('\n\n')
            lf_cont.write('Context orig:\n' + context_orig)
            lf_cont.write('\n\n')
            out_ids_docs = [str(id) + '\n' + doc for id, doc in zip(outlier_ids, outlier_docs)]
            lf_cont.write('Outlier docs:\n' + '\n'.join(out_ids_docs))
            lf_cont.write('\n\n')
            context_written = True
        lf_cont.close()