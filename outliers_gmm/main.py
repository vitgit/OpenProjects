import os

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from processing_methods import run_w_features_n_centroid
from utils import seed_everything, create_folder, get_random_questions_from_files, \
     qa_model_setup, create_chunks, create_db, \
     get_random_qa_from_files_SQUAD2, get_random_qa_from_files_mteb_trec_covid

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    #========================================================================
    cache_dir               = 'C:/hf_models'
    offload_folder          = 'C:/hf_models/offload'
    token_file              = base_dir + '/data/token_hugging_face.txt'
    # base_model_name         = 'mistralai/Mistral-7B-Instruct-v0.2'
    base_model_name         = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    tokenizer_model_name    = base_model_name

    log_dir                 = base_dir + '/log'

    index_path_SQUAD2       = 'C:/data/docs/Stanford_Question_Answering_Dataset/big_topics/db/faiss.index'
    chunks_file_SQUAD2      = 'C:/data/docs/Stanford_Question_Answering_Dataset/big_topics/db/chunks.json'
    questions_dir_SQUAD2    = 'C:/data/docs/Stanford_Question_Answering_Dataset/big_topics/questions'
    answers_dir_SQUAD2      = 'C:/data/docs/Stanford_Question_Answering_Dataset/big_topics/answers'
    docs_dir_SQUAD2         = 'C:/data/docs/Stanford_Question_Answering_Dataset/big_topics/docs'

    index_path_mteb_trec_covid      = 'C:/data/mteb-trec-covid_dataset/retrieved_standard/db/faiss.index'
    chunks_file_mteb_trec_covid     = 'C:/data/mteb-trec-covid_dataset/retrieved_standard/db/chunks.json'
    questions_dir_mteb_trec_covid   = 'C:/data/mteb-trec-covid_dataset/retrieved_standard/chatgpt_q_a_Q'
    answers_dir_mteb_trec_covid     = 'C:/data/mteb-trec-covid_dataset/retrieved_standard/chatgpt_q_a_A'
    docs_dir_mteb_trec_covid        = 'C:/data/mteb-trec-covid_dataset/retrieved_standard/docs'

    index_path_bloomberg_quint_news     = 'C:/data/docs/bloomberg_quint_news/db/faiss.index'
    chunks_file_bloomberg_quint_news    = 'C:/data/docs/bloomberg_quint_news/db/chunks.json'
    questions_dir_bloomberg_quint_news  = 'C:/data/docs/bloomberg_quint_news/chatgpt_q_a/questions'
    answers_dir_bloomberg_quint_news    = 'C:/data/docs/bloomberg_quint_news/chatgpt_q_a/answers'
    docs_dir_bloomberg_quint_news       = 'C:/data/docs/bloomberg_quint_news/docs'
    create_folder('C:/data/docs/bloomberg_quint_news/db')

    index_path_stanford_qa              = 'C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/db/fiass.index'
    chunks_file_stanford_qa             = 'C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/db/chanks.json'
    questions_dir_stanford_qa           = 'C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/questions'
    answers_dir_stanford_qa             = 'C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/answers'
    docs_dir_stanford_qa                = 'C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/docs'
    create_folder('C:/data/docs/Stanford_Question_Answering_Dataset/chatgpt_qa/db')


    # for general case ====================
    docs_path_gen           = 'C:/data/docs/bbc'
    questions_dir_gen       = 'C:/data/svd_experiments/questions'
    chunk_file_gen          = 'C:/data/svd_experiments/db/chunks/chunks.json'
    index_path_gen          = 'C:/data/svd_experiments/db/index/faiss.index'
    chunk_overlap           = 0
    # =====================================

    st_model_name           = 'sentence-transformers/all-mpnet-base-v2'
    st_cache_folder         = 'C:/hf_models'

    num_retrieved_docs      = 20
    max_new_tokens          = 1500

    num_questions           = -1 # if -1 all questions in files
    print_entire_responce   = False

    dataset_name            = 'stanford_qa' #'stanford_qa' 'bloomberg_quint_news'  #'mteb-trec-covid'   #'general' or 'SQUAD2' or 'mteb-trec-covid'
    force_db_creation       = False
    seed_everything_flag    = True
    use_custom_splitter     = False

    n_components_pca_list = [2, 3]  # [1, 2]
    n_clusters_gmm_list = [4, 5, 6]  # [2, 3, 4, 5]
    percentile = 15
    feature_method = 'interaction'  # 'concatenate', 'weighted_sum', 'interaction', 'polynomial'
    excluded_outlier_idxs = [0, 1, 2, 3]
    min_outlier_freq = 2
    alpha = 0.5
    '''
    # The bigger ALPHA the bigger weight to question distances, the closer to centroid.
    # To get more strength to query set the smaller ALPHA:
    '''
    #========================================================================
    create_folder(log_dir)
    if seed_everything_flag:
        seed_everything()

    base_model, tokenizer, llm_configs = qa_model_setup(base_model_name, tokenizer_model_name, max_new_tokens, cache_dir, offload_folder, token_file)

    st_model = SentenceTransformer(st_model_name, cache_folder=st_cache_folder)
    emb_dim = st_model.get_sentence_embedding_dimension()
    print('''
            max_seq_length                  = {}
            sentence_embedding_dimension    = {}
            '''.format(st_model.get_max_seq_length(), emb_dim))

    if dataset_name == 'SQUAD2':
        chunks_file             = chunks_file_SQUAD2
        index_path              = index_path_SQUAD2
        questions_dir           = questions_dir_SQUAD2
        answers_dir             = answers_dir_SQUAD2
        docs_path               = docs_dir_SQUAD2
        chunk_size              = 0
        chunk_overlap           = 0
        custom_text_separator   = '\n'
        qa_list = get_random_qa_from_files_SQUAD2(questions_dir, answers_dir, num_questions)
        questions = [item[0] for item in qa_list]

    elif dataset_name == 'general':
        chunks_file     = chunk_file_gen
        index_path      = index_path_gen
        questions_dir   = questions_dir_gen
        docs_path       = docs_path_gen
        questions = get_random_questions_from_files(questions_dir, num_questions)

    elif dataset_name == 'mteb-trec-covid':
        chunks_file = chunks_file_mteb_trec_covid
        index_path = index_path_mteb_trec_covid
        questions_dir = questions_dir_mteb_trec_covid
        answers_dir = answers_dir_mteb_trec_covid
        docs_path = docs_dir_mteb_trec_covid
        chunk_size = 0
        chunk_overlap = 0
        custom_text_separator = '\n'
        qa_list = get_random_qa_from_files_mteb_trec_covid(questions_dir, answers_dir, num_questions)
        questions = [item[0] for item in qa_list]

    elif dataset_name == 'bloomberg_quint_news':
        chunks_file             = chunks_file_bloomberg_quint_news
        index_path              = index_path_bloomberg_quint_news
        questions_dir           = questions_dir_bloomberg_quint_news
        answers_dir             = answers_dir_bloomberg_quint_news
        docs_path               = docs_dir_bloomberg_quint_news
        chunk_size              = 0
        chunk_overlap           = 0
        custom_text_separator   = '\n'
        qa_list = get_random_qa_from_files_mteb_trec_covid(questions_dir, answers_dir, num_questions, rand=False)
        questions = [item[0] for item in qa_list]
    elif dataset_name == 'stanford_qa':
        chunks_file = chunks_file_stanford_qa
        index_path = index_path_stanford_qa
        questions_dir = questions_dir_stanford_qa
        answers_dir = answers_dir_stanford_qa
        docs_path = docs_dir_stanford_qa
        chunk_size = 200
        chunk_overlap = 0
        custom_text_separator = '. ' #'\n'
        use_custom_splitter = True
        qa_list = get_random_qa_from_files_mteb_trec_covid(questions_dir, answers_dir, num_questions, rand=False)
        questions = [item[0] for item in qa_list]

    config = AutoConfig.from_pretrained(base_model_name)
    print('Configuration of ' + base_model_name + ':\n', config)

    if force_db_creation:
        chunked_documents = create_chunks(docs_path, chunks_file, chunk_size, chunk_overlap, custom_text_separator,
                                          tokenizer,
                                          use_custom_splitter=use_custom_splitter)
        embeddings = st_model.encode(chunked_documents, convert_to_tensor=False,
                                            show_progress_bar=True
                                            )
        print('embeddings shape', embeddings.shape)
        create_db(embeddings, index_path)

    run_w_features_n_centroid(questions, qa_list, dataset_name,
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
                              min_outlier_freq
                              )
