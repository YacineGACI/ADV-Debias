from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm
from scipy import stats
import tqdm


def sembias_evaluation(w2v):
    '''
    Evaluates word analogy using the SemBias dataset
    @w2v is the embedding model
    '''

    sembias_filepath = "data/evaluation/sembias/SemBias.tsv"

    print("*** SemBias evaluation ***")
    # Counts of most similar pair to "he - she" in the entire SemBias dataset
    gender_definitional_num = 0
    gender_stereoptype_num = 0
    none_num = 0
    total_num = 0

    # Counts of most similar pair to "he - she" in the subset SemBias dataset
    sub_gender_definitional_num = 0
    sub_gender_stereoptype_num = 0
    sub_none_num = 0
    sub_total_num = 0
    sub_start = 400

    gender_vector = w2v['he'] - w2v['she']

    with open(sembias_filepath, 'r') as f:
        for i, line in tqdm.tqdm(enumerate(f.readlines())):
            pairs = line.strip().split('\t')
            
            # Find the most similar pair difference to he-she
            max_similarity = -100
            max_index = -1
            for index, p in enumerate(pairs):
                p_1, p_2 = p.split(":")
                diff_vector = w2v[p_1] - w2v[p_2]
                similarity = dot(gender_vector, diff_vector)/(norm(gender_vector)*norm(diff_vector))

                if similarity > max_similarity:
                    max_similarity = similarity
                    max_index = index

            if max_index == 0:
                gender_definitional_num += 1
                if i >= sub_start:
                    sub_gender_definitional_num += 1
                
            elif max_index == 3:
                gender_stereoptype_num += 1
                if i >= sub_start:
                    sub_gender_stereoptype_num += 1

            else:
                none_num += 1
                if i >= sub_start:
                    sub_none_num += 1

            total_num += 1
            if i >= sub_start:
                sub_total_num += 1

    print('definition: {}'.format(gender_definitional_num / total_num))
    print('stereotype: {}'.format(gender_stereoptype_num / total_num))
    print('none: {}'.format(none_num / total_num))

    print('sub definition: {}'.format(sub_gender_definitional_num / sub_total_num))
    print('sub stereotype: {}'.format(sub_gender_stereoptype_num / sub_total_num))
    print('sub none: {}'.format(sub_none_num / sub_total_num))










def similarity_evaluation(w2v, dataset="ws"):
    '''
    Evaluates the similarity between pairs of words.
    @w2v @w2v is the embedding model
    @dataset is the benchmark dataset to use: [ws, rg, mturk, rw, men, simlex]
    '''

    def process_line(line):
        if dataset == "ws":
            _, w1, w2, sim = line.strip().split('\t')
        elif dataset == "rg":
            w1, w2, sim = line.strip().split(';')
        elif dataset == "mturk":
            w1, w2, sim = line.strip().split(',')
        elif dataset == "rw":
            w1, w2, sim, _ = line.strip().split('\t', 3)
        elif dataset == "men":
            w1, w2, sim = line.strip().split(' ')
        else:
            w1, w2, _, sim, _ = line.strip().split('\t', 4)
        
        return w1.lower(), w2.lower(), float(sim)


    print("*** {} Similarity evaluation ***".format(dataset.upper()))


    if dataset == "ws":
        evaluation_filepath = "data/evaluation/similarity/wordsim353.txt"
    elif dataset == "rg":
        evaluation_filepath = "data/evaluation/similarity/rubenstein_goodenough.csv"
    elif dataset == "mturk":
        evaluation_filepath = "data/evaluation/similarity/MTURK-771.csv"
    elif dataset == "rw":
        evaluation_filepath = "data/evaluation/similarity/rw.txt"
    elif dataset == "men":
        evaluation_filepath = "data/evaluation/similarity/MEN_dataset"
    else:
        evaluation_filepath = "data/evaluation/similarity/SimLex-999.txt"
    

    ground_truth = []
    predicted = []

    with open(evaluation_filepath, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            w1, w2, sim = process_line(line)

            try:
                predicted.append(w2v.similarity(w1, w2))
                ground_truth.append(sim)
            except KeyError as e:
                print(str(e))
    
    print("Spearman correlation: {}".format(stats.spearmanr(ground_truth, predicted)[0]))









if __name__ == "__main__":

    emb_filepath = "embeddings/glove.txt"
    emb_vectors = KeyedVectors.load_word2vec_format(emb_filepath, binary=False)


    sembias_evaluation(emb_vectors)
    print()
    similarity_evaluation(emb_vectors, dataset="ws")
    print()
    similarity_evaluation(emb_vectors, dataset="rg")
    print()
    similarity_evaluation(emb_vectors, dataset="mturk")
    print()
    similarity_evaluation(emb_vectors, dataset="men")
    print()
    similarity_evaluation(emb_vectors, dataset="simlex")
    print()