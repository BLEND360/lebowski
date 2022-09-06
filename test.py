from transformers.pipelines import pipeline

if __name__ == '__main__':
    # For testing, and forcing the download of the model
    pipeline('zero-shot-classification', model='facebook/bart-large-mnli')(
        'input', candidate_labels=['politics', 'news'], multi_label=True)
    pipeline('summarization', model='google/pegasus-cnn_dailymail')('input')
    pipeline('summarization', model='tuner007/pegasus_paraphrase')('input')
