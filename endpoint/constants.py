__all__ = ('MODEL_ARGS',)

MODEL_ARGS = {
    'sum1': (('summarization',), {
        'model': 'google/pegasus-cnn_dailymail'
    }),
    'sum2': (('summarization',), {
        'model': 'tuner007/pegasus_paraphrase'
    }),
    'zsc': (('zero-shot-classification',), {
        'model': 'facebook/bart-large-mnli'
    })
}
