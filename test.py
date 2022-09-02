from transformers.pipelines import pipeline

if __name__ == '__main__':
    # For testing, and forcing the download of the model
    pipeline('summarization', model='google/pegasus-cnn_dailymail')('input')
