import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Constants
# model_name = "sberbank-ai/mGPT"
model_name = "GroNLP/gpt2-medium-dutch-embeddings"

# Initialize model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

def solve_analogy(model, tokenizer, question, choices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_score = float('-inf')
    best_choice = None

    for choice in choices:
        input_text = question + " " + choice + "."
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        choice_ids = tokenizer.encode(choice, return_tensors="pt").view(-1)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -len(choice_ids):].cpu()
            softmaxes = logits.softmax(dim=2)
            choice_likelihoods = torch.gather(softmaxes, 2, choice_ids.unsqueeze(0).unsqueeze(2)).log()
            average_log_likelihood = choice_likelihoods.mean()

        if average_log_likelihood > max_score:
            max_score = average_log_likelihood
            best_choice = choice

    return best_choice

def clean_data(df):
    # copy correct response D to 'Ans5' so we can loop through response options
    df['Ans5'] = df['D']
    # drop unneeded cols
    drop_cols = ['row_number', 'type_verband', 'modified_count', 'item_rating', 'freq_corrC', 'freq_incorr1', 'freq_incorr2', 'freq_incorr3', 'freq_incorr4']
    df = df.drop(columns = drop_cols)
    # format col names
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.replace('-','', regex = True)
    df = df.replace("'", '', regex = True)
    # use only part of dataset for testing
    #df = df.head(6)
    return df

def add_template_sentences(df):
    templates = {
        "sentence1": "{} staat tot {}, zoals {} staat tot",
        "sentence2": "{} hoort bij {}, zoals {} hoort bij",
        "sentence3": "{} is vergelijkbaar met {}, zoals {} vergelijkbaar is met",
        "sentence4": "{} is tot {}, zoals {} is tot",
        "sentence5": "{} hoort bij {} op dezelfde manier dat {} hoort bij"
    }
    for key, template in templates.items():
        df[key] = df.apply(lambda x: template.format(x['A'], x['B'], x['C']), axis=1)
    return df

def filter_dataframe(df, tokenizer):
    df = df[df['Ans5'].apply(lambda x: len(tokenizer.encode(x)) == 1)]
    def count_multitoken_answers(row):
        count = sum(1 for i in range(1, 5) if len(tokenizer.encode(row[f'Ans{i}'])) > 1)
        return count
    df = df[df.apply(count_multitoken_answers, axis=1) <= 3]
    return df

def get_answers_from_dataframe(model, tokenizer, df):
    sentences = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
    all_results = {}
    for sentence_key in sentences:
        results = []
        for _, row in df.iterrows():
            question = row[sentence_key]
            choices = [row[f'Ans{i}'] for i in range(1, 6) if len(tokenizer.encode(row[f'Ans{i}'])) == 1]
            if not choices:
                results.append('None')
                continue
            answer = solve_analogy(model, tokenizer, question, choices)
            results.append(answer)
        df[sentence_key + "_answer"] = results
        all_results[sentence_key] = results
    return df, all_results

def compare_answers_to_Ans5(df, answers):
    for sentence_key, answers_list in answers.items():
        correctness_column = sentence_key + "_correctness"
        df[correctness_column] = df.apply(lambda row: row[sentence_key + "_answer"] == row['Ans5'], axis=1)
    return df

def calculate_mode_correctness(df):
    correctness_columns = [
        'sentence1_correctness', 'sentence2_correctness', 'sentence3_correctness',
        'sentence4_correctness', 'sentence5_correctness'
    ]
    df['mode_correctness'] = df[correctness_columns].mode(axis=1)[0].astype(bool)
    return df

def calculate_accuracy(df):
    correct_predictions = df[df['mode_correctness'] == True]
    return len(correct_predictions) / len(df)

if __name__ == "__main__":
    csv_file_path = '../data/prowise_verbal_analogy_data.csv'
    df = pd.read_csv(csv_file_path)

    df = clean_data(df)
    df = add_template_sentences(df)
    df = filter_dataframe(df, tokenizer)
    df, answers = get_answers_from_dataframe(model, tokenizer, df)
    df = compare_answers_to_Ans5(df, answers)
    df = calculate_mode_correctness(df)

    csv_output_path = 'data_ABC_prompt/gpt2_prompt_ABC_results.csv'
    df.to_csv(csv_output_path, index=False)

    accuracy = calculate_accuracy(df)
    print(f"Accuracy of the mode_correctness: {accuracy * 100:.2f}%")



