import streamlit as st

train_data = 'C:/Users/laptops.vn/Desktop/markov-predict-next-word-master/test_text.txt'
first_possible_words = {}
second_possible_words = {}
transitions = {}

def expandDict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

def get_next_probability(given_list):  
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict

def trainMarkovModel():
    for line in open(train_data):
        tokens = line.rstrip().lower().split()
        tokens_length = len(tokens)
        for i in range(tokens_length):
            token = tokens[i]
            if i == 0:
                first_possible_words[token] = first_possible_words.get(
                    token, 0) + 1
            else:
                prev_token = tokens[i - 1]
                if i == tokens_length - 1:
                    expandDict(transitions, (prev_token, token), 'END')
                if i == 1:
                    expandDict(second_possible_words, prev_token, token)
                else:
                    prev_prev_token = tokens[i - 2]
                    expandDict(
                        transitions, (prev_prev_token, prev_token), token)

    first_possible_words_total = sum(first_possible_words.values())
    for key, value in first_possible_words.items():
        first_possible_words[key] = value / first_possible_words_total

    for prev_word, next_word_list in second_possible_words.items():
        second_possible_words[prev_word] = get_next_probability(next_word_list)

    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = get_next_probability(next_word_list)

def next_word(tpl):
    if(type(tpl) == str):
        d = second_possible_words.get(tpl)
        if (d is not None):
            return list(d.keys())
    if(type(tpl) == tuple):  
        d = transitions.get(tpl)
        if(d == None):
            return []
        return list(d.keys())
    return None  

trainMarkovModel() 

def generate_next_word(seed_word):
    if seed_word:
        tkns = seed_word.split()
        if len(tkns) < 2:  
            suggestions = next_word(tkns[0].lower())
        else: 
            suggestions = next_word((tkns[-2].lower(), tkns[-1].lower()))
        return suggestions
    return []

def app():
    st.title("Next word Prediction ")
    st.write(
        "Enter the word, the program will make the next word prediction based on the Markov model.")
    seed_word = st.text_input("Input:", "")
    suggestions = generate_next_word(seed_word)

    if seed_word and suggestions:
        st.write("Next word:")
        st.write(suggestions)
    elif seed_word:
        st.warning("Can't find anymore.")

if __name__ == '__main__':
    app()
