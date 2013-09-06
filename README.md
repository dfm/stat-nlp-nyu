stat-nlp-nyu
============

Statistical Natural Language Processing at NYU

Python implementation
---------------------

To build the C-extension that computes edit distance, run

```
python setup.py build_ext --inplace
```

Then, to try the baseline model, run

```
python hw1_runner.py --data /path/to/data [--verbose]
```

To implement your own models, subclass `LanguageModel` in
`nlp/lang_models.py` and override the methods `__init__` and
`get_sentence_lnprobability`. If your model is called `MySickModel`, you would
run the model as

```
python hw1_runner.py --data /path/to/data --model MySickModel --verbose
```

If it takes any arguments, you can run:

```
python hw1_runner.py --data /path/to/data --model BigramModel --args "factor=0.9"
```
