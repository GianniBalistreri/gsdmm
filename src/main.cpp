//
// Created by Gianni Balistreri on 13.05.21.
//
#include <cmath>
#include <iostream>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

class GSDMM {
public:
    double alpha_;
    double beta_;
    int n_clusters_;
    int n_iterations_;
    int vocab_size_;
    int n_documents;
    std::vector<std::string> vocab_;
    std::vector<double> probability_vector;
    std::vector<int> cluster_word_count;
    std::vector<int> cluster_document_count;
    std::vector<std::map<std::string, int>> cluster_word_distribution;
    /*
    ############################
    # GSDMM Class Constructor: #
    ############################
    */
    GSDMM(std::vector<std::string> vocab, int vocab_size, int n_clusters, int n_iterations, double alpha, double beta) {
        alpha_ = alpha;
        beta_ = beta;
        n_clusters_ = n_clusters;
        n_iterations_ = n_iterations;
        vocab_size_ = vocab_size;
        vocab_ = vocab;
        n_documents = 0;
    }
    /*
    ##############
    # Fit model: #
    ##############
    */
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<std::map<std::string, int>>> fit(std::vector<std::vector<std::string>> documents){
        std::vector<double> probability_vector(n_clusters_, 1.0 / n_clusters_);
        std::vector<int> cluster_word_count(n_clusters_, 0);
        std::vector<int> cluster_document_count(n_clusters_, 0);
        std::map<std::string, int> vocab_count;
        for (std::string voc: vocab_){
            vocab_count.insert(std::pair<std::string, int>(voc, 0));
        }
        std::vector<std::map<std::string , int>> cluster_word_distribution(n_clusters_, vocab_count);
        n_documents = documents.size();
        std::vector<int> document_cluster(n_documents);
        // Allocate documents to cluster initially:
        int number_of_clusters = n_clusters_;
        int i = 0;
        for (auto doc = documents.begin(); doc != documents.end(); doc++){
            // Choose a random initial cluster for the document
            std::vector<double> probability_vector(n_clusters_, 1.0 / n_clusters_);
            int idx = sampling(probability_vector);
            document_cluster[i] = idx;
            cluster_document_count[idx] += 1;
            cluster_word_count[idx] += doc->size();
            for (std::string element : *doc){
                cluster_word_distribution[idx][element] += 1;
            }
            i += 1;
        }
        // Re-allocate documents to cluster iteratively:
        std::vector<double> probability_vector_;
        for (int iter = 0; iter < n_iterations_; iter++){
            int j = 0;
            int total_transfers = 0;
            for (auto doc = documents.begin(); doc != documents.end(); doc++){
                // Remove the document from it's current cluster:
                int old_cluster = document_cluster[j];
                cluster_document_count[old_cluster] -= 1;
                cluster_word_count[old_cluster] -= doc->size();
                for (std::string element : *doc){
                    cluster_word_distribution[old_cluster][element] -= 1;
                }
                // Score document:
                std::vector<double> p(n_clusters_, 0.0);
                double ld1 = log(n_documents - 1 + n_clusters_ * alpha_);
                int document_size = doc->size();
                for (int c = 0; c < n_clusters_; c++) {
                    double ln1 = log(cluster_word_count[c] + alpha_);
                    double ln2 = 0;
                    double ld2 = 0;
                    for (std::string word: documents[j]){
                        ln2 += log(cluster_word_distribution[c][word] + beta_);
                    }
                    for (int k = 0; k < document_size; k++){
                        ld2 += log(cluster_word_count[c] + vocab_size_ * beta_ + k - 1);
                    }
                    p[c] = exp(ln1 - ld1 + ln2 - ld2);
                }
                // Normalize probabilities:
                double probability_sum = 0;
                for (double norm_prob: p){
                    probability_sum += norm_prob;
                }
                if (probability_sum <= 0){
                    probability_sum = 1;
                }
                std::vector<double> probability(n_clusters_, 0.0);
                int n = 0;
                for (double prob: p){
                    double prob_each_cluster = prob / probability_sum;
                    probability[n] = prob_each_cluster;
                    n += 1;
                }
                try {
                    // Draw sample from multinomial distribution to find new cluster:
                    int new_cluster = sampling(probability);
                    if (new_cluster != old_cluster){
                        total_transfers += 1;
                    }
                    document_cluster[j] = new_cluster;
                    cluster_document_count[new_cluster] += 1;
                    cluster_word_count[new_cluster] += doc->size();
                    for (std::string element : *doc){
                        cluster_word_distribution[new_cluster][element] += 1;
                    }
                } catch (...) {
                    // Document stays in current cluster:
                    cluster_document_count[old_cluster] += 1;
                    cluster_word_count[old_cluster] += doc->size();
                    for (std::string element : *doc){
                        cluster_word_distribution[old_cluster][element] += 1;
                    }
                }
                j += 1;
            }
            double new_cluster_count = 0.0;
            for (auto doc_count = cluster_document_count.begin(); doc_count != cluster_document_count.end(); doc_count++){
                if (*doc_count > 0){
                    new_cluster_count += 1;
                }
            }
            if (total_transfers == 0 && new_cluster_count == number_of_clusters && iter > n_iterations_ - 5) {
                break;
            }
            number_of_clusters = (int) new_cluster_count;
        }
        n_clusters_ = number_of_clusters;
        return std::make_tuple(document_cluster, cluster_word_count, cluster_document_count, cluster_word_distribution);
    }
    /*
    ##########################
    # Predict cluster label: #
    ##########################
    int predict(std::vector<std::string> document){
        std::vector<double> probabilities = document_scoring(document);
        double highest_probability = 0.0;
        int i = 0;
        int cluster_label = 0;
        for (auto p = probabilities.begin(); p != probabilities.end(); p++){
            i += 1;
            if (*p > highest_probability){
                highest_probability = *p;
                cluster_label = i - 1;
            }
        }
        return cluster_label;
    }
    */
    /*
    ##########################################
    # Predict probabilities of each cluster: #
    ##########################################
    std::vector<double> predict_proba(std::vector<std::string> documents){
        std::vector<double> probabilities = document_scoring(documents);
        return probabilities;
    }
    */
    /*
    ####################################
    # Allocate cluster labels to text: #
    ####################################
    std::vector<int> generate_topic_allocation(std::vector<std::vector<std::string>> documents){
        std::vector<int> topic_allocation;
        for (auto doc = documents.begin(); doc != documents.end(); doc++){
            int cluster_label = predict(*doc);
            topic_allocation.push_back(cluster_label);
        }
        return topic_allocation;
    }
    */
    /*
    #################################
    # Get top-n words each cluster: #
    #################################
    std::map<std::string, std::vector<std::pair<std::string, int>>> get_top_words_each_cluster(int top_n_words) {
        std::map<std::string, std::vector<std::pair<std::string, int>>> top_words_each_cluster;
        for (int cluster_label = 0; cluster_label < cluster_word_distribution.size(); cluster_label++) {
            for (std::pair<std::string, int> element: cluster_word_distribution[cluster_label]) {
            //for (auto element: cluster_word_distribution[cluster_label]) {
                std::vector <std::pair<std::string, int>> vec;
                std::copy(vec.begin(), vec.end(), std::back_inserter<std::vector<std::pair<std::string, int>>>(vec));
                std::sort(vec.begin(), vec.end(),
                          [](const std::pair<std::string, int> &l, const std::pair<std::string, int> &r) {
                              if (l.second != r.second) {
                                  return l.second < r.second;
                              }
                              return l.first < r.first;
                          });
                std::map<std::string, int> subset;
                int i = 0;
                for (auto element: vec) {
                    if (i >= top_n_words && i <= vec.size()) {
                        subset.insert(std::pair<std::string, int>(element.first, element.second));
                    }
                    i += 1;
                }
                std::vector<std::pair<std::string, int>> subset_vec;
                std::copy(subset.begin(),
                          subset.end(),
                          std::back_inserter < std::vector < std::pair < std::string, int>>>(subset_vec));
                std::sort(subset_vec.begin(), subset_vec.end(),
                          [](const std::pair<std::string, int> &l, const std::pair<std::string, int> &r) {
                              if (l.second != r.second) {
                                  return l.second < r.second;
                              }
                              return l.first < r.first;
                          });
                std::reverse(subset_vec.begin(), subset_vec.end());
                top_words_each_cluster.insert(std::vector<std::pair<std::string, int>>(cluster_label, subset_vec));
            }
        }
        return top_words_each_cluster;
    }
    */
    /*
    #########################################
    # Get word importance for each cluster: #
    #########################################
    std::vector<std::pair<std::string , double>> word_importance_each_cluster(){
        std::vector<std::pair<std::string , double>> phi;
        for (int c = 0; c < n_clusters_; c++){
            for (std::pair<std::string, int> element: cluster_word_distribution[c]) {
                double sum_of_words = 0.0;
                for (std::pair<std::string, int> element: cluster_word_distribution[c]){
                    sum_of_words += element.second;
                }
                double importance = (element.second + beta_) / (sum_of_words + vocab_size_ + beta_);
                phi.insert(std::pair<std::string , double>(element.first, importance));
            }
        }
        return phi;
    }
    */
    /*
    ####################
    # Score documents: #
    ####################
    */
    std::vector<double> document_scoring(std::vector<std::string> document,
                                         double alpha,
                                         double beta,
                                         int n_clusters,
                                         int vocab_size,
                                         int n_documents,
                                         std::vector<int> cluster_word_count_,
                                         std::vector<int> cluster_document_count_,
                                         std::vector<std::map<std::string, int>> cluster_word_distribution_
                                         ){
        std::vector<double> p(n_clusters, 0.0);
        int document_size = document.size();
        double ld1 = log(n_documents - 1 + n_clusters * alpha);
        for (int i = 0; i < n_clusters; i++) {
            double ln1 = log(cluster_word_count_[i] + alpha);
            double ln2 = 0;
            double ld2 = 0;
            for (std::string word: document){
                ln2 += log(cluster_word_distribution_[i][word] + beta);
            }
            for (int j = 0; j < document_size; j++){
                ld2 += log(cluster_word_count_[i] + vocab_size * beta + j - 1);
            }
            p[i] = exp(ln1 - ld1 + ln2 - ld2);
        }
        // Normalize probabilities:
        double probability_sum = 0;
        for (double norm_prob: p){
            probability_sum += norm_prob;
        }
        if (probability_sum <= 0){
            probability_sum = 1;
        }
        std::vector<double> probability(n_clusters, 0.0);
        int n = 0;
        for (double prob: p){
            double prob_each_cluster = prob / probability_sum;
            probability[n] = prob_each_cluster;
            n += 1;
        }
        return probability;
    }
private:
    /*
    #########################
    # Multinomial Sampling: #
    #########################
    */
    // Sample index value representing cluster number from a multinomial distribution:
    int sampling(std::vector<double> probability_vector){
        py::array_t<float> probability_vector_py_obj = py::array_t<float>(py::cast(probability_vector));
        py::module np = py::module::import("numpy");
        auto sample_vector = np.attr("random").attr("multinomial")(1, probability_vector_py_obj).cast<std::vector<int>>();
        int idx = 0;
        for (auto i = sample_vector.begin(); i != sample_vector.end(); i++){
            if (*i == 1){
                break;
            } else {
                idx += 1;
            }
        }
        return idx;
    }
};

PYBIND11_MODULE(fast_gsdmm, m){
    py::class_<GSDMM>(m, "GSDMM")
    .def(py::init<std::vector<std::string>, int, int, int, double, double>())
    .def("fit", &GSDMM::fit)
    .def("document_scoring", &GSDMM::document_scoring);
    //.def("generate_topic_allocation", &GSDMM::generate_topic_allocation)
    //.def("get_top_words_each_cluster", &GSDMM::get_top_words_each_cluster)
    //.def("predict", &GSDMM::predict)
    //.def("predict_proba", &GSDMM::predict_proba)
    //.def("word_importance_each_cluster", &GSDMM::word_importance_each_cluster);
}