import random
import jieba
import numpy as np
import umap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cluster
import sys
import json
import os

# Load config path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KMVE_RG.config import config as args


def _preprocess_text(documents):
    cleaned_documents = [doc.lower().replace("\n", " ").replace("\t", " ") for doc in documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents


def get_item(cls, ann, sentence_all):
    examples = ann[cls]
    print(cls, '_len:', len(examples))
    for i, example in enumerate(examples):
        sentence = example['finding']
        sentence_all.append({f"{cls}_{i}": sentence})


def get_all_data(split, ann_path):
    ann = json.loads(open(ann_path, 'r', encoding='utf-8-sig').read())
    sentence_all = []
    for cls in split:
        get_item(cls, ann, sentence_all)
    print('sentence_alllen:', len(sentence_all))

    data = []
    cut_data = []
    for item in sentence_all:
        sentence = list(item.values())[0]
        data.append(sentence)
        cut_data.append(' '.join(jieba.lcut(sentence)))

    print('data:', len(data))
    return data, sentence_all, ann, cut_data


def _check_class_nums(topics, topic_model):
    cls_num = {}
    for item in topics:
        if item not in cls_num:
            cls_num[item] = str(item)

    result = len(cls_num) == topic_model.get_topic_info().shape[0]
    assert result is True, 'cls_nums need to equal topic_model.get_topic_info().shape'


def shuffle_result(topics, topic_model, ann, data, all_sentence, shuffle=False):
    _check_class_nums(topics, topic_model)
    all_data = []
    for i in range(len(data)):
        label = topics[i] + 1
        key_list = list(all_sentence[i].keys())[0].split('_')
        origin = ann[key_list[0]][int(key_list[1])]
        origin.update({'label': label})
        all_data.append(origin)
    if shuffle:
        random.shuffle(all_data)
        print('Shuffle data completed!')
    return all_data


if __name__ == '__main__':
    ann_path = args.ann_path
    split = ['train', 'val', 'test']

    # Load and preprocess data
    data, all_sentence, origin_ann, cut_data = get_all_data(split=split, ann_path=ann_path)

    documents = pd.DataFrame({
        "Document": data,
        "ID": range(len(data)),
        "Topic": None
    })

    # === Embedding (Options: SentenceTransformer or CountVectorizer) ===
    # from sentence_transformers import SentenceTransformer
    # embedding_method = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # embeddings = embedding_method.encode(data, show_progress_bar=True)

    count_vectorizer = CountVectorizer()
    embeddings = count_vectorizer.fit_transform(cut_data)

    # === Dimensionality Reduction ===
    umap_model = umap.UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        low_memory=False
    )

    umap_embeddings = umap_model.fit_transform(embeddings)
    new_embeddings = np.nan_to_num(umap_embeddings)

    # === Clustering ===
    # from hdbscan import HDBSCAN
    # model = HDBSCAN(min_cluster_size=10)

    model = cluster.KMeans(n_clusters=5, random_state=42)
    model.fit(new_embeddings)

    # Assign results
    labels = model.labels_
    documents['Topic'] = labels

    # Cluster summary
    sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
    topic_num = sizes.shape[0]
    topic_size = dict(zip(sizes.Topic, sizes.Document))

    print(f"Discovered {topic_num} topics.")
    print("Topic sizes:", topic_size)