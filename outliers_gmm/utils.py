import json
import os
import random
import re
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import faiss
import numpy as np
import torch
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import logger as text_splitter_logger
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from natsort import natsorted
from nltk.tokenize import word_tokenize
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from features import create_distance_features, transform_and_plot

class count_tokens_stat:
    num_tokens_tot = 0
    num_tokens_max = 0
    num_tokens_min = 10000000
    num_tokens_av = 0.
    times_limit_exceeded = 0

    def __init__(self, text, num, limit):
        nn = count_tokens(text)
        count_tokens_stat.num_tokens_tot += nn
        if count_tokens_stat.num_tokens_max < nn:
            count_tokens_stat.num_tokens_max = nn
        if count_tokens_stat.num_tokens_min > nn:
            count_tokens_stat.num_tokens_min = nn
        count_tokens_stat.num_tokens_av = round(float(count_tokens_stat.num_tokens_tot) / num)
        if nn > limit:
            count_tokens_stat.times_limit_exceeded += 1
def print_documents_stat(stat):
        print(
'''
Custom Splitter:
Number of tokens: {}
Max number of tokens in chunk: {}
Min number of tokens in chunk: {}
Average number of tokens in chunk: {}
Number of documents after custom split: {}
Number of times chunk_size exceeded: {} \n 
'''.format(stat[0], stat[1], stat[2], stat[3], stat[4], stat[5])
        )
def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)

def count_documents_tokens_stat(documents, limit):
    doc_count = 0
    for doc in documents:
        doc_count += 1
        if isinstance(doc, Document):
            text = doc.page_content
        else:
            text = doc
        count_tokens_stat(text, doc_count, limit)

def create_db(embeddings, index_path):
    dimension = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dimension)  # Use IndexFlatL2 for Euclidean distance
    index.add(embeddings)
    faiss.write_index(index, index_path)
def retrieve_vectors_by_ids(index_path, ids):
    index = faiss.read_index(index_path)
    retrieved_vectors = []
    for id in ids:
        # Faiss IndexFlatL2 supports the reconstruct method
        vector = np.zeros((index.d), dtype='float32')
        index.reconstruct(int(id), vector)
        retrieved_vectors.append(vector)
    return np.array(retrieved_vectors)
def normalize_vec(x):
    return x / np.linalg.norm(x)
def encode(x, model, normalize= False):
    y = model.encode(x, convert_to_tensor=False, show_progress_bar=True)
    if normalize:
        y = normalize_vec(y)
    return y
def create_folder(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)
    print(dir, 'has been created or checked')
def db_search_from_index(question, model_name, cache_folder, index_file, chunks_file, num_retrieved_docs):
    query_embedding = convert_docs_to_embeddings([question], model_name, cache_folder)

    docs_list, dist_list, indexes = search_from_index(index_file, query_embedding, num_retrieved_docs, chunks_file)
    return docs_list, dist_list, query_embedding, indexes
def convert_docs_to_embeddings(documents, model_name, cache_folder, ret_type='ndarray'):
    model = SentenceTransformer(model_name, cache_folder=cache_folder)

    if isinstance(documents[0], Document):
        sentences = [doc.page_content for doc in documents]
    elif isinstance(documents[0], str):
        sentences = documents

    embeddings_tensor = model.encode(sentences, convert_to_tensor=True)
    if ret_type == 'tensor':
        return embeddings_tensor
    else:
        return np.array(embeddings_tensor)
def search_from_index(index_file, query_embedding, num_retrieved_docs, chunks_file):
    faiss_index = faiss.read_index(index_file)
    D, I = faiss_index.search(query_embedding, num_retrieved_docs)

    with open(chunks_file) as json_file:
        docs_dic = json.load(json_file)
    docs_list = []
    # get docs:
    for ii in I[0]:
        docs_list.append(docs_dic[str(ii)])
    return docs_list, list(D[0]), list(I[0])
def read_key_from_file(file):
    f = open(file, 'r')
    string = f.read()
    f.close()
    return string
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_random_questions_from_files(dir, num_q):
    questions_set = set()
    for filename in os.listdir(dir):
        file = os.path.join(dir, filename)
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf8') as f:
                questions = f.readlines()
                for q in questions:
                    q = q.strip()
                    questions_set.add(q)
    if num_q < 0:
        return list(questions_set)
    else:
        return random.choices(list(questions_set), k=num_q)
def get_random_qa_from_files_SQUAD2(q_dir, a_dir, num_q, rand=True):

    q_files_list = sorted(os.listdir(q_dir))
    a_files_list = sorted(os.listdir(a_dir))
    if len(q_files_list) != len(a_files_list):
        print('len(q_files_list) != len(a_files_list)')
        sys.exit(1)

    qa_set = set()
    for ii in range(len(q_files_list)):
        file_q = os.path.join(q_dir, q_files_list[ii])
        file_a = os.path.join(a_dir, a_files_list[ii])
        fq = open(file_q, 'r', encoding='utf8')
        fa = open(file_a, 'r', encoding='utf8')
        questions   = fq.readlines()
        answers     = fa.readlines()
        fq.close()
        fa.close()
        for q, a in zip(questions, answers):
            qa_set.add((q, a))
    if num_q < 0:
        return get_questions_w_answers(list(qa_set))
    else:
        return random.choices(get_questions_w_answers(qa_set), k=num_q)

def get_random_qa_from_files_mteb_trec_covid(q_dir, a_dir, num_q, rand=True):
    q_files_list = natsorted(os.listdir(q_dir))
    a_files_list = natsorted(os.listdir(a_dir))
    if num_q < 0:
        num_q = len(q_files_list)
    qa_list = []
    for ii in range(len(q_files_list)):
        file_q = os.path.join(q_dir, q_files_list[ii])
        file_a = os.path.join(a_dir, a_files_list[ii])
        with open(file_q, 'r', encoding='utf8') as fq:
            q = fq.read().strip()
        with open(file_a, 'r', encoding='utf8') as fa:
            a = fa.read().strip()
        qa_list.append((q, a))
    if rand:
        return random.choices(qa_list, k=num_q)
    else:
        return [qa for ii, qa in enumerate(qa_list) if ii < num_q]

def qa_model_setup(model_name, tokenizer_model_name, max_new_tokens, cache_dir, offload_folder, token_file, repetition_penalty=1.1, do_sample=False):
    llm_configs = {
        # 'model':model,
        # 'tokenizer':tokenizer,
        # 'task':"text-generation",
        # 'temperature': 0.0,       # 0 for consistent reproducible answers, default is 0
        'repetition_penalty': repetition_penalty,
        # 'return_full_text':True,
        'max_new_tokens': max_new_tokens,     # Set's the possible response length
        'do_sample': do_sample,         # If True - enables stochastic or probabilistic text generation
                                    # If False - deterministically picks the most likely next token at each step of the generation
    }

    token = read_key_from_file(token_file)
    if 'mistral' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    cache_dir=cache_dir,
                                                    device_map="auto",
                                                    local_files_only=False,  # True,
                                                    token=token,
                                                    offload_folder=offload_folder,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch.float32
                                                    ).to(torch.device("cpu"))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name,
                                                  cache_dir=cache_dir,
                                                  token=token,
                                                 )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)

    return model, tokenizer, llm_configs

def get_response_tinyllama(tokenizer, question, context, model, max_new_tokens=150):
    num_words_in_prompt = count_tokens(question + ' ' + context)
    print('num_words_in_prompt:', num_words_in_prompt)

    formatted_prompt = f'''
    You are a friendly chatbot who responds to the user's question by looking into context.</s>

    Context: 
{context}
</s>

    Question: {question}</s>
    '''
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, padding=True)

    # Generate the response
    start_time = time.time()

    outputs = model.generate(**inputs,
                             # max_length=max_new_tokens,
                             max_new_tokens=max_new_tokens)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for LLM response: {elapsed_time:.4f} seconds")

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pattern = re.compile(r"Answer:(.*)(?:\n|$)", re.DOTALL)
    if len(pattern.findall(response)) < 1:
        return 'Not Found', '_', '_'
    response1 = '\n<Answer:>' + pattern.findall(response)[0]
    return response1, num_words_in_prompt, str(elapsed_time)
def get_response(tokenizer, prompt, model, llm_configs):
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(inputs.input_ids, **llm_configs)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
def num_files_in_folder(dir):
    onlyfiles = next(os.walk(dir))[2]
    return len(onlyfiles)
def extract_text_by_page_with_pdfminer(pdf_path):
    with open(pdf_path, 'rb') as file:
        resource_manager = PDFResourceManager()
        fake_file_handle = StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        documents = []
        page_num = 0
        for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
            page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

            doc = Document(page_content = text, metadata = {'source': pdf_path, 'page': str(page_num)})
            page_num += 1
            documents.append(doc)

            # Clear the StringIO object for the next page
            fake_file_handle.truncate(0)
            fake_file_handle.seek(0)

        converter.close()
        fake_file_handle.close()

        return documents


class CustomCharacterTextSplitter(CharacterTextSplitter):
    def __init__(self, custom_splitter, max_chunk_size, custom_separator, tokenizer,
                 # tokenizer_name="muntasir2179/TinyLlama-1.1B-rag-finetuned-v1.0",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_splitter = custom_splitter
        self.max_chunk_size = max_chunk_size
        self.custom_separator = custom_separator
        self.tokenizer = tokenizer  #AutoTokenizer.from_pretrained(tokenizer_name)

    def split_text(self, text):
        # Use the default splitting method
        chunks = super().split_text(text)

        # Check if any chunk exceeds the max_chunk_size
        final_chunks = []
        for chunk in chunks:
            tokenized_chunk = self.tokenizer(chunk, return_tensors="pt")
            num_tokens = tokenized_chunk['input_ids'].shape[1]
            if num_tokens > self.max_chunk_size:
                # Apply the custom splitter with the custom separator
                final_chunks.extend(self.custom_splitter(chunk, self.custom_separator))
            else:
                final_chunks.append(chunk)

        return final_chunks

# Example custom splitter function
def my_custom_splitter(text, separator):
    return text.split(separator)

def create_chunks(docs_path, chunk_file, chunk_size, chunk_overlap, custom_text_separator, tokenizer,
                  use_pdfminer=True, use_custom_splitter=False):
    num_files = num_files_in_folder(docs_path)
    print(docs_path + ' contains ' + str(num_files) + ' files')
    if num_files < 1:
        print('\nNo files in ' + docs_path)
        sys.exit(1)
    t1 = datetime.now()
    documents = []
    # Create a List of Documents from all of our files in the ./docs folder
    files = [os.path.join(docs_path, file) for file in os.listdir(docs_path)]
    # Sort files by time modified
    files.sort(key=os.path.getmtime)
    for file in files:
        if file.endswith(".pdf"):
            if use_pdfminer:
                documents.extend(extract_text_by_page_with_pdfminer(file))
            else:
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(file, encoding='utf-8')
            documents.extend(loader.load())
    # Chunk text
    if use_custom_splitter:
        splitter = CustomCharacterTextSplitter(
            custom_splitter=my_custom_splitter,
            max_chunk_size=chunk_size,  # Set your max chunk size
            custom_separator=custom_text_separator,  # Set your custom separator
            chunk_size=chunk_size,  # Set the initial chunk size
            chunk_overlap=chunk_overlap,  # Set the chunk overlap
            tokenizer=tokenizer
        )

        text = '\n'.join([doc.page_content for doc in documents])
        chunked_documents = splitter.split_text(text)
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=custom_text_separator
        )
        print('\nStarting Text Splitter ..... \n')
        # suppress warning for chunk > chunk_size
        text_splitter_logger.disabled = True
        chunked_documents = text_splitter.split_documents(documents)
        text_splitter_logger.disabled = False

    print('\nText Splitter done\n')

    num_docs = len(chunked_documents)
    count_documents_tokens_stat(chunked_documents, chunk_size)
    stat = (count_tokens_stat.num_tokens_tot, count_tokens_stat.num_tokens_max, count_tokens_stat.num_tokens_min,
            count_tokens_stat.num_tokens_av, num_docs, count_tokens_stat.times_limit_exceeded)
    print_documents_stat(stat)

    if isinstance(chunked_documents[0], str):
        dict_cunks = {str(index): line for index, line in enumerate(chunked_documents)}
    else:
        dict_cunks = {str(index): line.page_content for index, line in enumerate(chunked_documents)}

    with open(chunk_file, 'w') as f:
        json.dump(dict_cunks, f)
    return chunked_documents
def get_questions_w_answers(qa_list):
    new_qas = []
    for qa in qa_list:
        if not 'No Answer' in qa[1]:
            new_qas.append(qa)
    return new_qas
def simple_prompt(question, context):
    return f'''
[INST]
Answer below Question using below Context. Mark it <Answer:>
[/INST]

[Question]
{question}

[Context]
{context}
'''

def emb_cosine_similarity(sent1, sent2, st_model):
    emb1 = st_model.encode(sent1.strip(), convert_to_tensor=True)
    emb2 = st_model.encode(sent2.strip(), convert_to_tensor=True)
    cosine_score = util.cos_sim(emb1, emb2)
    cosine_score = cosine_score.numpy()[0][0]
    return round(float(cosine_score), 3)
def tfidf_cosine_similarity(sentence1, sentence2):
    # Vectorizer to convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    # Transform sentences to get TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([sentence1.strip(), sentence2.strip()])
    # Compute and return the cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]), 3)
def contains_similarity_boolean(sent1, sent2):
    s1 = sent1.lower().strip()
    s2 = sent2.lower().strip()
    if (s1 in s2) or (s2 in s1):
        return True
    else:
        return False
#=========class similarity_stat_count===========================================================
class SimilarityStatCount:
    def __init__(self):
        # Initialize instance-level attributes
        self.count = 0
        self.contains_count_1 = 0
        self.contains_count_2 = 0
        self.sim_emb_1_better_than_2_count = 0
        self.sim_emb_2_better_than_1_count = 0
        self.sim_tfidf_1_better_than_2_count = 0
        self.sim_tfidf_2_better_than_1_count = 0
        self.sum_reduction = 0.0
        self.sum_explained_variance = 0.0
        self.sum_emb_sim_1 = 0.0
        self.sum_emb_sim_2 = 0.0
        self.sum_tfidf_sim_1 = 0.0
        self.sum_tfidf_sim_2 = 0.0
        self.sum_num_filtered_docs = 0
    def do_count(self):
        self.count += 1
        return self.count
    def count_similarities(self, sent1, sent2, sent_ground_truth, st_model=None):

        if st_model != None:
            s1 = emb_cosine_similarity(sent1, sent_ground_truth, st_model)
            s2 = emb_cosine_similarity(sent2, sent_ground_truth, st_model)
            if s1 >= s2: self.sim_emb_1_better_than_2_count += 1
            if s2 >= s1: self.sim_emb_2_better_than_1_count += 1

        if contains_similarity_boolean(sent1, sent_ground_truth): self.contains_count_1 += 1
        if contains_similarity_boolean(sent2, sent_ground_truth): self.contains_count_2 += 1

        s1 = tfidf_cosine_similarity(sent1, sent_ground_truth)
        s2 = tfidf_cosine_similarity(sent2, sent_ground_truth)
        if s1 >= s2: self.sim_tfidf_1_better_than_2_count += 1
        if s2 >= s1: self.sim_tfidf_2_better_than_1_count += 1
    def count_av_reduction(self, proj_size, red_size, count):
        self.sum_reduction += float(red_size)/float(proj_size)
        av_reduction = self.sum_reduction / float(count)
        return av_reduction
    def get_av_explained_variance(self, explained_variance):
        self.sum_explained_variance += explained_variance
        return float(self.sum_explained_variance / self.count)
    def get_av_num_filtered_docs(self, num_filtered_docs):
        self.sum_num_filtered_docs += num_filtered_docs
        return float(self.sum_num_filtered_docs / self.count)
    def get_av_similarities(self, sent1, sent2, sent_ground_truth, st_model):
        self.sum_emb_sim_1      += emb_cosine_similarity(sent1, sent_ground_truth, st_model)
        self.sum_emb_sim_2      += emb_cosine_similarity(sent2, sent_ground_truth, st_model)
        self.sum_tfidf_sim_1    += tfidf_cosine_similarity(sent1, sent_ground_truth)
        self.sum_tfidf_sim_2    += tfidf_cosine_similarity(sent2, sent_ground_truth)
        return \
            [
            self.sum_emb_sim_1      / float(self.count),
            self.sum_emb_sim_2      / float(self.count),
            self.sum_tfidf_sim_1    / float(self.count),
            self.sum_tfidf_sim_2    / float(self.count)
            ]
    def get_av_sim_change(self):
        return \
        float((self.sum_emb_sim_2   - self.sum_emb_sim_1)   / self.sum_emb_sim_1), \
        float((self.sum_tfidf_sim_2 - self.sum_tfidf_sim_1) / self.sum_tfidf_sim_1)
    def get_counts(self):
        return (self.contains_count_1,
                self.contains_count_2,
                self.sim_emb_1_better_than_2_count,
                self.sim_emb_2_better_than_1_count,
                self.sim_tfidf_1_better_than_2_count,
                self.sim_tfidf_2_better_than_1_count
                )
    def print_counts(self, mode='print'):

        to_print = f'''
contains_count_1:                   {self.contains_count_1}
contains_count_2:                   {self.contains_count_2}
sim_emb_1_better_than_2_count:      {self.sim_emb_1_better_than_2_count}
sim_emb_2_better_than_1_count:      {self.sim_emb_2_better_than_1_count}
sim_tfidf_1_better_than_2_count:    {self.sim_tfidf_1_better_than_2_count}
sim_tfidf_2_better_than_1_count:    {self.sim_tfidf_2_better_than_1_count}        
'''
        if mode == 'string':
            return to_print
        else:
            print(to_print)
#==========class similarity_stat_count=========================================================

def print_similarities(sent1, sent2, sent_ground_truth, st_model):
    sim_emb_1 = emb_cosine_similarity(sent1, sent_ground_truth, st_model)
    sim_emb_2 = emb_cosine_similarity(sent2, sent_ground_truth, st_model)
    sim_tfidf_1 = tfidf_cosine_similarity(sent1, sent_ground_truth)
    sim_tfidf_2 = tfidf_cosine_similarity(sent2, sent_ground_truth)

    to_print = f'''
sim_emb_1:      {sim_emb_1}
sim_emb_2:      {sim_emb_2}
sim_tfidf_1:    {sim_tfidf_1}
sim_tfidf_2:    {sim_tfidf_2}
'''
    return to_print

def extract_clusters_from_vector_features(vectors, ids, query_vector, alpha, plot_file,
                                          n_components_pca_list=[2],
                                          n_clusters_gmm_list=[2],
                                          min_outlier_freq=0,
                                          percentile=10,
                                          feature_method='concatenate',
                                          excluded_outlier_idxs=[]
                                          ):
    centroid = np.mean(vectors, axis=0)

    features = create_distance_features(vectors, centroid, query_vector, alpha, method=feature_method, degree=2)

    # Standardize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    print("Shape of combined normalized features:", features_normalized.shape)

    outlier_id_freq = {}
    print('>>>>> getting outliers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for n_clusters_gmm in n_clusters_gmm_list:
        for n_components_pca in n_components_pca_list:
            print('n_components_pca = ', n_components_pca, '  n_clusters_gmm = ', n_clusters_gmm)

            # Apply PCA for dimensionality reduction if there are at least 2 features
            if n_components_pca <= min(features_normalized.shape[0], features_normalized.shape[1]):
                pca = PCA(n_components=n_components_pca)
                features_pca = pca.fit_transform(features_normalized)
            else:
                features_pca = features_normalized  # Skip PCA if there is only one feature

            # Apply Gaussian Mixture Models (GMM) for clustering
            gmm = GaussianMixture(n_components=n_clusters_gmm, covariance_type='full', random_state=42)
            gmm.fit(features_pca)
            labels = gmm.predict(features_pca)

            # Calculate the likelihood of each point
            log_likelihood = gmm.score_samples(features_pca)

            # Identify outliers based on low likelihood
            threshold = np.percentile(log_likelihood, percentile)  # Consider the bottom 10% as outliers
            outlier_ids = [ids[i] for i in range(len(ids)) if log_likelihood[i] < threshold]

            print('\nOutliers on this iteration:')
            for ii, id in enumerate(ids):
                aa = str(ii) + ')  ' + str(id)
                if id in outlier_ids:
                    print(aa, '\t- outlier')
                else:
                    print(aa)

            for id in outlier_ids:
                if len(outlier_ids) < 1:
                    continue
                if id in outlier_id_freq.keys():
                    outlier_id_freq[id] += 1
                else:
                    outlier_id_freq[id] = 1

            title = 'ALPHA = ' + str(alpha) + ' n_clusters_gmm = ' + str(n_clusters_gmm) + '  feature method = ' + feature_method
            transform_and_plot(vectors, ids, labels, log_likelihood, threshold, centroid, query_vector, alpha=0.5,
                               method='concatenate', degree=2, plot_file=plot_file, title=title)

            print('>>>>> outliers done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    outlier_ids_most_freq = [id for id in outlier_id_freq.keys() if outlier_id_freq[id] >= min_outlier_freq and
                             id not in [ids[ii] for ii in excluded_outlier_idxs]]
    print('\noutlier_ids_most_freq with ALPHA = ' + str(alpha) + '   min_outlier_freq = ' + str(min_outlier_freq) +
          '   feature method = ' + feature_method + '  excluded_outlier_idxs = ' + str(excluded_outlier_idxs) + ':')
    for ii, id in enumerate(ids):
        aa = str(ii) + ')  ' + str(id)
        if id in outlier_ids_most_freq:
            print(aa, '\t- outlier')
        else:
            print(aa)

    return outlier_ids_most_freq

def output_wrapper(docs, vectors, ids, query_vector, alpha, plot_file, n_components_pca_list, n_clusters_gmm_list,
                   min_outlier_freq, percentile, feature_method, excluded_outlier_idxs):
    outlier_ids = extract_clusters_from_vector_features(vectors, ids, query_vector, alpha, plot_file,
                                                        n_components_pca_list,
                                                        n_clusters_gmm_list,
                                                        min_outlier_freq,
                                                        percentile,
                                                        feature_method,
                                                        excluded_outlier_idxs
                                                       )
    remaining_docs  = [doc for id, doc in zip(ids, docs) if id not in outlier_ids]
    remaining_ids   = [id for id in ids if id not in outlier_ids]
    outlier_docs    = [doc for id, doc in zip(ids, docs) if id in outlier_ids]
    return remaining_docs, remaining_ids, None, outlier_docs, outlier_ids

