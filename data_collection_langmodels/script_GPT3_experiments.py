'''
Script for retrieving the GPT-3 verbal analogy completions.
File includes data pre-processing utilities and evaluation functions.
Results are written to a seperate csv file.
'''
import sys
import argparse
import csv
import numpy as np
import pandas as pd
import itertools
import random
import os
import openai

# get starttime to compute runtime
from datetime import datetime
start = datetime.now()

# 
sys.path.append("..")

# load API key from environment variable 
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "PUT YOUR OPENAI API KEY HERE"

def clean_data(df):
    # copy correct response D to 'Ans5' so we can loop through response options
    df['Ans5'] = df['D']
    # format col names
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.replace('-','', regex = True)
    df = df.replace("'", '', regex = True)
    # use only part of dataset for testing
    #df = df.head(6)
    print(df.head())
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

def data_prep(path_to_file, type_prompts, num_templates, num_ans_options):
    df = pd.read_csv(path_to_file)
    df = clean_data(df)
    df = add_template_sentences(df, type_prompts)
    
    return df

# function to get gpt-4 responses to analogy multiple choice questions
def ask_chatgpt_mc(model_engine, question, Ans1, Ans2, Ans3, Ans4, Ans5):
    options = [Ans1, Ans2, Ans3, Ans4, Ans5]
    # shuffle options so that the correct option is not always the last one
    random.shuffle(options)
    
    prompt = f"{question}? Kies {options[0]}, {options[1]}, {options[2]}, {options[3]} of {options[4]}."
    
    response = openai.ChatCompletion.create(
      model = model_engine,
      messages = [
            {"role": "user", "content": prompt}
        ],
        max_tokens = 20,
        temperature = 0
    )
    #print(response['choices'][0])
    
    return response['choices'][0]['finish_reason'], response['choices'][0]['message']['content']

# function to use instead of gpt request to test programming logic
def ask_gpt_test(engine, prompt, ans):
    response = 'test response'
    finish_reason = 'test finish'
    first_token_index = len(prompt)
    d_avg_logprob = first_token_index
    
    return response, finish_reason, d_avg_logprob

# function to get gpt-3 responses to analogy multiple choice questions
def ask_gpt3_mc(model_engine, question, Ans1, Ans2, Ans3, Ans4, Ans5):
    options = [Ans1, Ans2, Ans3, Ans4, Ans5]
    # shuffle options so that the correct option is not always the last one
    random.shuffle(options)
    
    prompt = f"{question}? Kies {options[0]}, {options[1]}, {options[2]}, {options[3]} of {options[4]}."
    
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=20,
        n=1,
        temperature=0, # no randomization in response
        #logprobs=1 # request log probabilities for each output token
    )
    #print(response['choices'][0]["text"])
    #first_token_index = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) <= len(prompt))[0][-1]
    #d_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_index:])
    
    return response['choices'][0]['finish_reason'], response["choices"][0]["text"]

# function to get log prob of an analogy completion from gpt-3
def get_gpt3_completion_logprobs(model_engine, question, ans):
    prompt = f"{question} "
    d_prompt = prompt + f"{ans}"
    
    response = openai.Completion.create(
        engine=model_engine,
        prompt=d_prompt,
        max_tokens=len(d_prompt),
        n=1,
        stop=['.', '\n'],
        echo=True,
        temperature=0, # no randomization in response
        logprobs=1 # request log probabilities for each token in the input prompt
    )
    
    #print(response['choices'][0])
    first_token_idx = np.where(np.array(response['choices'][0]['logprobs']['text_offset']) <= len(prompt))[0][-1]
    d_avg_logprob = np.mean(response['choices'][0]['logprobs']['token_logprobs'][first_token_idx:])
    
    return d_avg_logprob, response['choices'][0]['finish_reason'], response["choices"][0]["text"]

def main():
    parser = argparse.ArgumentParser()
    
    # FILE FOR MAIN DATA COLLECTION
    parser.add_argument("--path_to_input_file", default='../items/prowise_verbal_analogy_items.csv', type=str, help="Path to data directory")
    # FILE FOR EXPERIMENT 4 DATA COLLECTION
    #parser.add_argument("--path_to_input_file", default='../items/dat_ACB_new_items.csv', type=str, help="Path to data directory")
    
    # MODIFY THESE TO CHANGE MODEL
    #parser.add_argument("--model_name_or_path", default='gpt-4-0613', type=str, help="Pretrained model path")
    parser.add_argument("--model_name_or_path", default='text-davinci-003', type=str, help="Pretrained model path")
    parser.add_argument("--model_shortname", default='gpt3', type=str, help="short name to refer to model")
    
    # ABC is for main data collection, the other ones are for experiments 2-4 
    parser.add_argument("--type_prompts", default='ABC', type=str, help="Type of prompts: ABC = A:B::C:?, C = C:?, ACB = A:C::B:?, B = B:?")
    
    # Directory to store output in 
    parser.add_argument("--output_dir", default='data/', type=str, help="Output directory for results")
    # Its good to store ACB new data in a different dir as otherwise it overwrites the ACB data in experiment 3
    #parser.add_argument("--output_dir", default='data_ACB_new_prompt/', type=str, help="Output directory for results")
    
    # fixed to 5 for this study
    parser.add_argument("--num_templates", default=5, type=int, help="Number of templates to average over")
    # update this is script times out
    parser.add_argument("--rowid_start", default=0, type=int, help="Rowid to start with if script times out, first row indexed at 0")
    args = parser.parse_args()
    
    # prep dataframe containing analogy items
    df = data_prep(args.path_to_file, args.type_prompts, args.num_templates, args.num_ans_options)
    
    # open csv files to write gpt responses and conversation log to, create writer and logger
    f = open(f'{args.output_dir}/{args.model_shortname}_prompt_{args.type_prompts}_results_{args.rowid_start}.csv', 'w')
    writer = csv.writer(f)
    # write headers to csv files 
    header = ['rowid', 'item_number', 'template_nr', 'answer_nr', 'finish_reason', 'response', 'logprob']
    writer.writerow(header)
    
    # start with item from rowid_start
    rowid = args.rowid_start
    num_items = len(df) 
    
    # get answers for each analogy in df
    for i in range(0, num_items - rowid):
        # get logprobs of each verbal analogy completion in rowid
        for j in range(1, args.num_templates + 1):
            # get result for sentence j on item i in dataframe
            if (args.model_shortname == 'gpt4'):
                finish_reason, response_text = ask_chatgpt_mc(args.model_name_or_path, df["sentence{0}".format(j)][rowid + i], 
                            df["Ans1"][rowid + i], df["Ans2"][rowid + i], df["Ans3"][rowid + i], df["Ans4"][rowid + i], df["Ans5"][rowid + i])
                # create row
                # note: 999 is missing value for answer_nr as we evaluate all answers in one go with chatgpt
                # note: we supply correct response in logprob column to ease scoring
                row = [rowid + i, df['item_number'][rowid + i], j, 999, finish_reason, response_text, df["Ans5"][rowid + i]]
                # write to csv
                writer.writerow(row)
            elif (args.model_shortname == 'gpt3'):
                finish_reason, response_text = ask_gpt3_mc(args.model_name_or_path, df["sentence{0}".format(j)][rowid + i], 
                            df["Ans1"][rowid + i], df["Ans2"][rowid + i], df["Ans3"][rowid + i], df["Ans4"][rowid + i], df["Ans5"][rowid + i])
                row = [rowid + i, df['item_number'][rowid + i], j, 999, finish_reason, response_text, df["Ans5"][rowid + i]]
                # write to csv
                writer.writerow(row)
            else:
                pass
            '''
            # removed to get gpt3 mc responses
                for k in range(1, args.num_ans_options + 1):
                    # get logprob of answer k on sentence j on item i
                    if (args.model_shortname == 'gpt3'):
                        logprob, finish_reason, response_text = get_gpt3_completion_logprobs(args.model_name_or_path, df["sentence{0}".format(j)][rowid + i], df["Ans{0}".format(k)][rowid + i])
                    else: 
                        logprob, finish_reason, response_text = ask_gpt_test(args.model_name_or_path, df["sentence{0}".format(j)][rowid + i], df["Ans{0}".format(k)][rowid + i])
                    # create data row and write to csv
                    row = [rowid + i, df['item_number'][rowid + i], j, k, finish_reason, response_text, logprob]
                    writer.writerow(row)
            '''
    # close csv writer and file
    f.close()
    

if __name__ == "__main__":

    main()
