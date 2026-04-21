#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

class BertTokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    int cls_token_id = 101; // [CLS] indicates start of sentence
    int sep_token_id = 102; // [SEP] indicates end of sentence
    int pad_token_id = 0;   // [PAD] fills empty space
    int unk_token_id = 100; // [UNK] used for unknown words
    int max_length_ = 12;   // Matches our DistilBERT Tensor size

public:
    // 1. Load the Vocabulary into memory
    BertTokenizer(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            std::cerr << "CRITICAL ERROR: Could not open " << vocab_path << "\n";
            exit(1);
        }
        std::string line;
        int index = 0;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back(); // Clean Windows line endings
            vocab_[line] = index++;
        }
        std::cout << "Tokenizer initialized with " << vocab_.size() << " words." << std::endl;
    }

    // 2. Translate English to Numbers
    void encode(std::string text, std::vector<int64_t>& input_ids, std::vector<int64_t>& attention_mask) {
        input_ids.clear();
        attention_mask.clear();

        // Convert string to lowercase
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);

        // Start every sequence with [CLS]
        input_ids.push_back(cls_token_id);
        attention_mask.push_back(1);

        // Split sentence by spaces
        std::stringstream ss(text);
        std::string word;
        while (ss >> word && input_ids.size() < max_length_ - 1) {
            // Strip punctuation safely
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            
            // Map word to Vocab ID
            if (vocab_.count(word)) {
                input_ids.push_back(vocab_[word]);
            } else {
                input_ids.push_back(unk_token_id);
            }
            attention_mask.push_back(1); // 1 means "pay attention to this real word"
        }

        // End sequence with [SEP]
        if (input_ids.size() < max_length_) {
            input_ids.push_back(sep_token_id);
            attention_mask.push_back(1);
        }

        // Pad the rest of the array with 0s
        while (input_ids.size() < max_length_) {
            input_ids.push_back(pad_token_id);
            attention_mask.push_back(0); // 0 means "ignore this empty padding"
        }
    }
};