import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from helpermethods import preprocessData
from protoNN_example import main


def convert_test_train(x_train, y_train, x_test, y_test):
    # print("shape1", y_train.shape)
    # print("shape2", x_train.shape)
    # print(y_train)
    # print(x_train)
    train = np.concatenate([y_train, x_train], axis=1)

    test = np.concatenate([y_test, x_test], axis=1)

    return train, test


def evaluate_protoNN(model, train_data, test_data, device, batch_size, checkpoint, proj_dim, num_proj):
    model.eval()

    with torch.no_grad():
        dim1 = len(train_data)
        dim2 = len(test_data)

        print (dim1)
        print (dim2)

        x_train=np.zeros(shape=(dim1,768))
        y_train=np.zeros(shape=(dim1,1))
        x_test=np.zeros(shape=(dim2,768))
        y_test=np.zeros(shape=(dim2,1))

        ind=0
        for pointer in tqdm(range(0, len(test_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(test_data): break
                sents = test_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(test_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_test[ind]=output_embedding[idx].detach().numpy()
                y_test[ind]=true_labels[idx].item()
                ind=ind+1

        ind=0
        train_start_ts = time.time()
        for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(train_data): break
                sents = train_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(train_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_train[ind]=output_embedding[idx].detach().numpy()
                # print(x_train[ind])
                y_train[ind]=true_labels[idx].item()
                ind=ind+1

    train, test = convert_test_train(x_train, y_train, x_test, y_test)
    _, _, x_train, y_train, x_test, y_test = preprocessData(train, test)

    train, test = convert_test_train(x_train, y_train, x_test, y_test)
    acc = main(train, test, device, proj_dim, num_proj)

    return acc


def evaluate_knn(model, train_data, test_data, device, batch_size, checkpoint, n_neighbors=5):
    model.eval()

    with torch.no_grad():
        dim1 = len(train_data)
        dim2 = len(test_data)

        print (dim1)
        print (dim2)

        x_train=np.zeros(shape=(dim1,768))
        y_train=np.zeros(shape=(dim1))
        x_test=np.zeros(shape=(dim2,768))
        y_test=np.zeros(shape=(dim2))

        ind=0
        for pointer in tqdm(range(0, len(test_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(test_data): break
                sents = test_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(test_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_test[ind]=output_embedding[idx].detach().numpy()
                y_test[ind]=true_labels[idx].item()
                ind=ind+1

        ind=0
        train_start_ts = time.time()
        for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(train_data): break
                sents = train_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(train_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_train[ind]=output_embedding[idx].detach().numpy()
                # print(x_train[ind])
                y_train[ind]=true_labels[idx].item()
                ind=ind+1

    print("Running PCA now")

    pca = PCA(n_components=70)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # print("PCA complete. Variance is", pca.explained_variance_ratio_)

    neigh = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
    print("Fitting KNN")
    neigh.fit(x_train, y_train)
    train_end_ts = time.time()

    print("Predicting KNN")
    predict_start_ts = time.time()
    out = neigh.predict(x_test)
    predict_end_ts = time.time()

    corr = 0
    total = 0
    for idx in range(dim2):
        total += 1
        if out[idx] == y_test[idx]:
            corr += 1

    acc = corr / total
    print("Accuracy KNN:", acc, "Training time:", train_end_ts-train_start_ts, "Predicting time:", predict_end_ts-predict_start_ts)

    return acc


def evaluate_svm(model, train_data, test_data, device, batch_size, checkpoint):
    model.eval()

    with torch.no_grad():
        dim1 = len(train_data)
        dim2 = len(test_data)

        print (dim1)
        print (dim2)

        x_train=np.zeros(shape=(dim1,768))
        y_train=np.zeros(shape=(dim1))
        x_test=np.zeros(shape=(dim2,768))
        y_test=np.zeros(shape=(dim2))

        ind=0
        for pointer in tqdm(range(0, len(test_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(test_data): break
                sents = test_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(test_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_test[ind]=output_embedding[idx].detach().numpy()
                y_test[ind]=true_labels[idx].item()
                ind=ind+1

        train_start_ts = time.time()
        ind=0
        for pointer in tqdm(range(0, len(train_data), batch_size),desc='training'):
            model.train() # model was in eval mode in evaluate(); re-activate the train mode
            # torch.cuda.empty_cache() # releases all unoccupied cached memory

            sent_pairs = []
            labels = []
            for i in range(pointer, pointer+batch_size):
                if i >= len(train_data): break
                sents = train_data[i].get_texts()
                if len(word_tokenize(' '.join(sents))) > 300: continue
                sent_pairs.append(sents)
                labels.append(train_data[i].get_label())
            logits, output_embedding = model.ff(sent_pairs,checkpoint)

            output_embedding = output_embedding.type(torch.FloatTensor)
            if logits is None: continue
            true_labels = torch.LongTensor(labels)

            for idx in range(output_embedding.shape[0]):
                x_train[ind]=output_embedding[idx].detach().numpy()
                # print(x_train[ind])
                y_train[ind]=true_labels[idx].item()
                ind=ind+1

    print("Running PCA now")

    pca = PCA(n_components=50)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # print("PCA complete. Variance is", pca.explained_variance_ratio_)

    clf = svm.SVC(kernel='rbf', C=0.01, cache_size=1000)
    print("Training SVM")
    clf.fit(x_train,y_train)
    train_end_ts = time.time()

    print("Predicting SVM")
    predict_start_ts = time.time()
    out = clf.predict(x_test)
    predict_end_ts = time.time()

    corr = 0

    # print("x_train", x_train)
    # print("y_train", y_train)
    # print("out", out)
    # print("y_test", y_test)
    # print("x_test", x_test)

    # conf=np.zeros(shape=(4,4))
    for i in range(x_test.shape[0]):
        if (out[i] == y_test[i]):
            corr=corr+1
        # conf[int(out[i])][int(y_test[i])]=conf[int(out[i])][int(y_test[i])]+1

    acc = corr/x_test.shape[0]

    print("Accuracy SVM:", acc, "Training time:", train_end_ts-train_start_ts, "Predicting time:", predict_end_ts-predict_start_ts)

    return acc


