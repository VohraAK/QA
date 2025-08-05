import collections
import evaluate
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm

squad_metric = evaluate.load("squad")
squadv2_metric = evaluate.load("squad_v2")

def preprocess_squad_training(data, tokenizer: AutoTokenizer, doc_stride):
    '''Preprocessing function which maps questions to answer spans in the context.'''
    
    tokenized_data = tokenizer(
        [q.strip() if isinstance(q, str) else "" for q in data["question"]],
        data["context"],
        max_length=tokenizer.model_max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    
    answers = data["answers"]
    offsets = tokenized_data.pop("offset_mapping")
    sample_maps = tokenized_data.pop("overflow_to_sample_mapping")  # maps where each overflowed feature corresponds to which example point
    
    starts = []
    ends = []
    
    for i, offset in enumerate(offsets):
        sample_i = sample_maps[i]   # get original example point index
        answer = answers[sample_i]  # get input answer
        
        # SQuAD v2 check
        if len(answer["text"]) <= 0:
            starts.append(0)
            ends.append(0)
            continue
        
        start_c = answer["answer_start"][0]
        end_c = start_c + len(answer["text"][0])
        
        seq_ids = tokenized_data.sequence_ids(i) 
        
        pos = 0
        
        # 1. find the starting position of the context (denoted by 1s)
        while pos < len(seq_ids) and seq_ids[pos] != 1:
            pos += 1
        
        # sanity check
        if seq_ids[pos] != 1:
            # question is too long, context is cut off in this feature
            starts.append(0)
            ends.append(0)
            continue
        
        context_start = pos
        
        # 2. find the ending position of the context
        while pos < len(seq_ids) and seq_ids[pos] == 1:
            pos += 1
        context_end = pos - 1
        
        
        # we now need to refine the context positions to span the answer.
        # first, check if the current offsets in the context contain the answer or not
        if offset[context_start][0] > start_c or offset[context_end][1] < end_c:
            starts.append(0)
            ends.append(0)
            continue
        else:
            # refining the window, staring from start of the context
            pos = context_start
            
            while pos <= context_end and offset[pos][0] <= start_c:
                pos += 1
            
            # loop breaks at the first instance of the offset being > start_c
            starts.append(pos - 1)

            pos = context_end
            
            while pos >= context_start and offset[pos][1] >= end_c:
                pos -= 1
            
            ends.append(pos + 1)
    
    tokenized_data["start_positions"] = starts
    tokenized_data["end_positions"] = ends
    
    return tokenized_data
        
def preprocess_squad_validation(data, tokenizer: AutoTokenizer, doc_stride):
    '''Special preprocessig function for validation data - maps created features to original examples via ID'''
    
    tokenized_data = tokenizer(
        [q.strip() if isinstance(q, str) else "" for q in data["question"]],
        data["context"],
        max_length=tokenizer.model_max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )

    sample_maps = tokenized_data.pop("overflow_to_sample_mapping")
    example_ids = []
    
    for i in range(len(tokenized_data["input_ids"])):
        sample_i = sample_maps[i]
        example_ids.append(data["id"][sample_i])
        
        seq_ids = tokenized_data.sequence_ids(i)
        offset = tokenized_data["offset_mapping"][i]
        
        tokenized_data["offset_mapping"][i] = [ofs if seq_ids[j] == 1 else None for j, ofs in enumerate(offset)]    # delete all question offsets
        
    tokenized_data["example_id"] = example_ids
    return tokenized_data


from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30, metric=squadv2_metric):
    start_logits = np.array(start_logits.detach().cpu())
    end_logits = np.array(end_logits.detach().cpu())
    
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
