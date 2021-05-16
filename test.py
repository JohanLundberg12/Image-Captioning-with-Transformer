import pandas as pd
import sys
import tensorflow as tf
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer


from evaluation import bleu1, bleu2, bleu3, bleu4, evaluate
from helper_functions import get_checkpoint_path, load_config
from load_data import extract_captions, extract_image_names, sample_from_df
from preprocessing import remove_unk
from pretrained_transformers import get_embedding, get_tokenizer
from transformer import build_transformer, CustomSchedule


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = load_config(config_file)
    config.checkpoint_path = get_checkpoint_path(config)
    
    tokenizer = get_tokenizer(config.lang)
    embedding = get_embedding(config.embedding_type, config, tokenizer)
    transformer = build_transformer(config, embedding, tokenizer)

    df_test = pd.read_csv(f'data/{config.lang}_test.csv')
    captions_test = extract_captions(df_test)
    img_names_test = extract_image_names(df_test, config.image_path)

    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

    # Loading Checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, config.checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    ckpt_path = tf.train.get_checkpoint_state(f'./checkpoints/{config.lang}/{config.embedding_type}/32/ckpt-1/').all_model_checkpoint_paths[14] #ckpt-15 best checkpoint
    ckpt.restore(ckpt_path)
    
    captions = []
    real_captions = []
    for caption in captions_test:
        caption = caption.rsplit(" ", 1)[0]
        real_captions.append(caption)

    for image in img_names_test:
        caption, result, attention_weights = evaluate(image, tokenizer, transformer)
        caption = remove_unk(caption, tokenizer.unk_token)
        caption = ' '.join(caption)
        caption = caption.rsplit(" ", 1)[0]
        captions.append(caption)

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_L_scores = []
    rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


    for reference, candidate in list(zip(real_captions, captions)):
        rouge_L_score = rouge_scorer.score(reference, candidate)
        rouge_L_scores.append(rouge_L_score['rougeL'].fmeasure)
        meteor_score = single_meteor_score(reference, candidate)
        meteor_scores.append(meteor_score)

        bleu1_scores.append(bleu1(reference, candidate))
        bleu2_scores.append(bleu2(reference, candidate))
        bleu3_scores.append(bleu3(reference, candidate))
        bleu4_scores.append(bleu4(reference, candidate))

    avg_bleu1_score = sum(bleu1_scores)/len(bleu1_scores)
    avg_bleu2_score = sum(bleu2_scores)/len(bleu2_scores)
    avg_bleu3_score = sum(bleu3_scores)/len(bleu3_scores)
    avg_bleu4_score = sum(bleu4_scores)/len(bleu4_scores)
    avg_rouge_L_score = sum(rouge_L_scores)/len(rouge_L_scores)
    avg_meteor_score = sum(meteor_scores)/len(meteor_scores)

    print(f"AVG BLEU-1 score: {avg_bleu1_score}")
    print(f"AVG BLEU-2 score: {avg_bleu2_score}")
    print(f"AVG BLEU-3 score: {avg_bleu3_score}")
    print(f"AVG BLEU-4 score: {avg_bleu4_score}")
    print(f"AVG Rouge L score: {avg_rouge_L_score}")
    print(f"AVG Meteor Score: {avg_meteor_score}")
    
    

