'''
Script for retrieving the normalized MLM analogy completions.
File includes data pre-processing utilities and evaluation functions.
All results are written back to a seperate csv file.
'''
import sys
import torch
import argparse
import numpy as np
import pandas as pd

from transformers import AutoModelForMaskedLM, AutoTokenizer

'''
list of BERT models used
#RobBERT 'pdelobelle/robbert-v2-dutch-base'
#Bertje 'GroNLP/bert-base-dutch-cased'
#MBERT 'bert-base-multilingual-cased'
#XLM-V 'facebook/xlm-v-base'
'''
# 
sys.path.append("..")

def clean_data(df):
    # copy correct response D to 'Ans5' so we can loop through response options
    df['Ans5'] = df['D']
    # format col names
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.replace('-','', regex = True)
    df = df.replace("'", '', regex = True)
    # use only part of dataset for testing
    #df = df.head(6)
    return df

def filter_out_multitoken_words(df, tokenizer):

    def count_id(word):
        # Count the number of token ids for a single word
        ans_space = " " + word
        tokenized_word = tokenizer.encode(ans_space, add_special_tokens=False)
        removed_brackets = ", ".join(str(item) for item in tokenized_word)
        count_id = len(removed_brackets.split())
        if count_id > 1:
            return "0"
        return word

    # Apply function count_id on all answeroptions, to get a zero on the answeroptions with more than one id
    df[['Ans1', 'Ans2', 'Ans3', 'Ans4', 'Ans5']] = df[['Ans1', 'Ans2', 'Ans3', 'Ans4', 'Ans5']].applymap(count_id)
    # Drop row if ans5 (correct answer) is zero, so more than one 1 id
    df.drop(df[df.Ans5 == "0"].index, inplace=True)
    # Drop rows if 3 or more alternative answeroptions are 0
    df["zero_rows"] = np.sum(df.loc[:, ['Ans1', 'Ans2', 'Ans3', 'Ans4']] == "0", axis=1)
    df = df[df['zero_rows'] < 4]
    df.drop("zero_rows", 1)
    return df

def add_template_sentences(df, type_prompts):
    # testing condition: A:B::C:?
    if (type_prompts == 'ABC'):
        df["sentence1"] = df.apply(lambda x: f"{x['A']} staat tot {x['B']}, zoals {x['C']} staat tot", axis=1)
        df["sentence2"] = df.apply(lambda x: f"{x['A']} hoort bij {x['B']}, zoals {x['C']} hoort bij", axis=1)
        df["sentence3"] = df.apply(lambda x: f"{x['A']} is vergelijkbaar met {x['B']}, zoals {x['C']} vergelijkbaar is met", axis=1)
        df["sentence4"] = df.apply(lambda x: f"{x['A']} is tot {x['B']}, zoals {x['C']} is tot", axis=1)
        df["sentence5"] = df.apply(lambda x: f"{x['A']} hoort bij {x['B']} op dezelfde manier dat {x['C']} hoort bij", axis=1)
    # testing condition: C:?
    elif (type_prompts == 'C'):
        df["sentence1"] = df.apply(lambda x: f"{x['C']} staat tot", axis=1)
        df["sentence2"] = df.apply(lambda x: f"{x['C']} hoort bij", axis=1)
        df["sentence3"] = df.apply(lambda x: f"{x['C']} is vergelijkbaar met", axis=1)
        df["sentence4"] = df.apply(lambda x: f"{x['C']} is tot", axis=1)
        df["sentence5"] = df.apply(lambda x: f"{x['C']} hoort op dezelfde manier bij", axis=1)
    # testing condition: A:C::B:?
    elif (type_prompts == 'ACB'):
        df["sentence1"] = df.apply(lambda x: f"{x['A']} staat tot {x['C']}, zoals {x['B']} staat tot", axis=1)
        df["sentence2"] = df.apply(lambda x: f"{x['A']} hoort bij {x['C']}, zoals {x['B']} hoort bij", axis=1)
        df["sentence3"] = df.apply(lambda x: f"{x['A']} is vergelijkbaar met {x['C']}, zoals {x['B']} vergelijkbaar is met", axis=1)
        df["sentence4"] = df.apply(lambda x: f"{x['A']} is tot {x['C']}, zoals {x['B']} is tot", axis=1)
        df["sentence5"] = df.apply(lambda x: f"{x['A']} hoort bij {x['C']} op dezelfde manier dat {x['B']} hoort bij", axis=1)
    # testing condition: B:?
    elif (type_prompts == 'B'):
        df["sentence1"] = df.apply(lambda x: f"{x['B']} staat tot", axis=1)
        df["sentence2"] = df.apply(lambda x: f"{x['B']} hoort bij", axis=1)
        df["sentence3"] = df.apply(lambda x: f"{x['B']} is vergelijkbaar met", axis=1)
        df["sentence4"] = df.apply(lambda x: f"{x['B']} is tot", axis=1)
        df["sentence5"] = df.apply(lambda x: f"{x['B']} hoort op dezelfde manier bij", axis=1)
    # else use type_prompt = ABC
    else:
        df = add_template_sentences(df, 1)
    return df

def data_prep(path_to_file, tokenizer, type_prompts):
    df = pd.read_csv(path_to_file)
    df = clean_data(df)
    df = filter_out_multitoken_words(df, tokenizer)
    df = add_template_sentences(df, type_prompts)
    return df

def load_model(model_name_or_path):
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, tokenizer

def solved_correct(x):
    max_value = max(x)
    # fifth element is the correct solution to the analogy
    fifth_element = x[4]
    if max_value == fifth_element:
        return True
    else:
        return False

# get MLM logits for specific [MASK] token index
def mlm_output(model, encoded_input, mask_token_index, ax):
    token_logits = model(encoded_input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    sm = torch.nn.Softmax(dim=ax)
    probs = sm(mask_token_logits)
    return probs

# get probabilities per template
def compute_probs(model, tokenizer, sentence, answers):
    encoded_input = tokenizer.encode(sentence + " " + tokenizer.mask_token + ".", return_tensors="pt")
    mask_token_index = torch.where(encoded_input == tokenizer.mask_token_id)[1]
    
    probs = mlm_output(model, encoded_input, mask_token_index, ax=1)
    
    result = []
    for ans in answers:
        ans_id = tokenizer.encode(" " + ans, add_special_tokens=False)
        prob = probs[0][ans_id].item()
        result.append(prob)
    
    return result

# get MLM output
def get_mlm_output(df, model, tokenizer):
    # store correct per response to calculate acc
    correct = []
    # number of templates used (this was 5)
    num_temps = 5
    
    # for each item in dataset
    for ix, row in df.iterrows():
        answers = [row[f'Ans{str(i)}'] for i in range(1,num_temps+1)] 
        # for each template
        for i in range(1, num_temps+1):
            # get probabilities for each answer
            results = compute_probs(model, tokenizer, row[f'sentence{str(i)}'], answers)
            row[f'prob_sentence{str(i)}'] = results
            row[f'result_sentence{str(i)}'] = solved_correct(results)
        # get correct predictions (mode gives the value that appears the most)
        df_mode = row[['result_sentence1', 'result_sentence2', 'result_sentence3', 'result_sentence4', 'result_sentence5']]
        correct.append(df_mode.mode().values[0])
    
    acc = np.round(correct.count(True)/len(correct)*100, 2)
    print(f'Accuracy: {acc}')
    return df, correct

def main():
    parser = argparse.ArgumentParser()
    
    # FILE FOR MAIN DATA COLLECTION
    parser.add_argument("--path_to_input_file", default='../items/prowise_verbal_analogy_items_selection.csv', type=str, help="Path to data directory")
    # FILE FOR EXPERIMENT 4 DATA COLLECTION
    #parser.add_argument("--path_to_input_file", default='../items/dat_ACB_new_items.csv', type=str, help="Path to data directory")
    
    # MODIFY THESE TO CHANGE MODEL
    parser.add_argument("--model_name_or_path", default='facebook/xlm-v-base', type=str, help="Pretrained model path")
    parser.add_argument("--model_shortname", default='xlmv', type=str, help="short name to refer to model")
    
    # ABC is for main data collection, the other ones are for experiments 2-4 
    parser.add_argument("--type_prompts", default='ABC', type=str, help="Type of prompts: ABC = A:B::C:?, C = C:?, ACB = A:C::B:?, B = B:?")
    
    # Directory to store output in 
    parser.add_argument("--output_dir", default='data/', type=str, help="Output directory for results")
    # Its good to store ACB new data in a different dir as otherwise it overwrites the ACB data in experiment 3
    #parser.add_argument("--output_dir", default='data_ACB_new_prompt/', type=str, help="Output directory for results")
    args = parser.parse_args()
    

    # data collection
    model, tokenizer = load_model(args.model_name_or_path)
    df = data_prep(args.path_to_input_file, tokenizer, args.type_prompts)
    df, correct = get_mlm_output(df, model, tokenizer)
    df[f'result_{str(args.model_shortname)}'] = correct
    
    # save output to csv file
    df.to_csv(f'{args.output_dir}/{args.model_shortname}_prompt_{args.type_prompts}_results.csv', index=False)
    
if __name__ == "__main__":

    main()
