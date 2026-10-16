"""
questions.py
============

Viva-voce question bank for the AI Autonomous Viva-Voce & Technical Examiner System.

Structure
---------
``QUESTION_BANK`` maps a technical subject (domain) to a list of question
entries. Every entry contains:

* ``id``            - unique identifier, e.g. ``"AIML-01"``
* ``question``      - the viva question put to the student
* ``ideal_answer``  - comprehensive model answer used as the semantic reference
* ``keywords``      - mandatory technical terms the examiner expects to hear
* ``topic``         - short topic label shown in the dashboard
* ``difficulty``    - "Foundational", "Intermediate" or "Advanced"

Keyword syntax
--------------
A keyword may list accepted alternatives separated by ``|`` - for example
``"global interpreter lock|gil"``. The first alternative is the display name.
Matching is case-insensitive and tolerant of plurals, verb forms, hyphenation
and British/American spelling (handled in ``evaluator.py``).
"""

from __future__ import annotations

from typing import Dict, Iterator, List

QUESTION_BANK: Dict[str, List[Dict[str, object]]] = {
    # =========================================================================
    # DOMAIN 1: AI & MACHINE LEARNING
    # =========================================================================
    "AI & Machine Learning": [
        {
            "id": "AIML-01",
            "topic": "Overfitting",
            "difficulty": "Foundational",
            "question": "What is overfitting in machine learning, and how can it be prevented?",
            "ideal_answer": (
                "Overfitting occurs when a machine learning model learns the noise and random "
                "fluctuations in the training data instead of the underlying pattern. Such a model "
                "shows very high accuracy on the training set but performs poorly on unseen test or "
                "validation data, which means it fails to achieve good generalization. Overfitting "
                "usually happens when the model complexity is too high relative to the amount of "
                "training data, for example a very deep decision tree or a neural network with too "
                "many parameters. It can be detected by comparing training error with validation "
                "error; a large gap between them indicates overfitting. Common prevention techniques "
                "include collecting more training data, using cross-validation to tune "
                "hyperparameters, applying regularization such as L1 (Lasso) or L2 (Ridge) penalties, "
                "reducing model complexity through pruning or feature selection, using dropout in "
                "neural networks, early stopping when the validation loss stops improving, and data "
                "augmentation. Ensemble methods such as bagging and random forests also reduce "
                "variance and therefore reduce overfitting."
            ),
            "keywords": [
                "training data",
                "noise",
                "generalization|generalise|generalize",
                "validation",
                "model complexity|complex model",
                "regularization",
                "cross-validation",
                "dropout",
                "early stopping",
            ],
        },
        {
            "id": "AIML-02",
            "topic": "Bias-Variance Tradeoff",
            "difficulty": "Intermediate",
            "question": "Explain the bias-variance tradeoff with an example.",
            "ideal_answer": (
                "The bias-variance tradeoff describes the balance between two sources of prediction "
                "error in supervised learning. Bias is the error caused by overly simple assumptions in "
                "the model; a high-bias model suffers from underfitting because it cannot capture the "
                "true relationship between the features and the target, for example fitting a straight "
                "line with linear regression to clearly non-linear data. Variance is the error caused "
                "by sensitivity to small fluctuations in the training data; a high-variance model "
                "suffers from overfitting because it memorizes noise, and its predictions change a lot "
                "when it is trained on a different sample, as with a very deep decision tree. The "
                "expected test error can be decomposed into bias squared, variance and irreducible "
                "error, which is the noise that no model can remove. Increasing model complexity "
                "usually lowers bias but raises variance, while simplifying the model does the "
                "opposite, so the goal is to find the optimal complexity that minimizes total error on "
                "unseen data. Cross-validation, regularization, ensemble methods such as bagging, which "
                "reduces variance, and boosting, which reduces bias, together with collecting more "
                "data, help manage the tradeoff."
            ),
            "keywords": [
                "bias",
                "variance",
                "underfitting",
                "overfitting",
                "model complexity",
                "irreducible error",
                "test error|total error",
                "regularization",
                "ensemble|bagging|boosting",
            ],
        },
        {
            "id": "AIML-03",
            "topic": "Learning Paradigms",
            "difficulty": "Foundational",
            "question": (
                "Differentiate between supervised, unsupervised and reinforcement learning, "
                "giving one application of each."
            ),
            "ideal_answer": (
                "Supervised learning trains a model on labeled data, where every input example is "
                "paired with the correct output, and the model learns a mapping from features to the "
                "target. It is used for classification tasks such as spam detection and regression "
                "tasks such as house price prediction, with algorithms like linear regression, "
                "logistic regression, decision trees and support vector machines. Unsupervised "
                "learning works with unlabeled data and discovers hidden structure or patterns on its "
                "own. Typical tasks are clustering, for example customer segmentation with k-means, "
                "dimensionality reduction with principal component analysis, and anomaly detection. "
                "Reinforcement learning is different from both: an agent interacts with an "
                "environment, takes actions and receives a reward or penalty, and through trial and "
                "error it learns a policy that maximizes the cumulative reward, as in game playing, "
                "robotics and self-driving systems. In short, supervised learning learns from correct "
                "answers, unsupervised learning learns from the structure of the data, and "
                "reinforcement learning learns from feedback on its actions."
            ),
            "keywords": [
                "labeled data|labelled data|labels",
                "unlabeled data|unlabelled data",
                "classification",
                "regression",
                "clustering",
                "dimensionality reduction",
                "agent",
                "environment",
                "reward",
                "policy",
            ],
        },
        {
            "id": "AIML-04",
            "topic": "Gradient Descent",
            "difficulty": "Intermediate",
            "question": "How does gradient descent work, and what is the role of the learning rate?",
            "ideal_answer": (
                "Gradient descent is an iterative optimization algorithm used to minimize a loss "
                "function, also called a cost function, by updating model parameters such as weights "
                "and biases. At each step it computes the gradient, which is the vector of partial "
                "derivatives of the loss with respect to each parameter, and moves the parameters in "
                "the opposite direction of the gradient because that is the direction of steepest "
                "descent. The size of each step is controlled by the learning rate hyperparameter: if "
                "the learning rate is too large the algorithm can overshoot the minimum and diverge, "
                "and if it is too small convergence becomes very slow. Batch gradient descent uses the "
                "whole dataset for every update, stochastic gradient descent uses a single sample, and "
                "mini-batch gradient descent uses small batches, which balances speed and stability. "
                "Improved optimizers such as momentum, RMSProp and Adam adapt the step size during "
                "training. For convex functions gradient descent reaches the global minimum, while for "
                "the non-convex loss of a neural network it may settle in a local minimum or a saddle "
                "point."
            ),
            "keywords": [
                "optimization|optimisation|minimize",
                "loss function|cost function",
                "gradient",
                "learning rate",
                "parameters|weights",
                "convergence|converge",
                "stochastic gradient descent|sgd",
                "mini-batch",
                "local minimum|global minimum",
            ],
        },
        {
            "id": "AIML-05",
            "topic": "Backpropagation",
            "difficulty": "Advanced",
            "question": "Explain how backpropagation trains a neural network.",
            "ideal_answer": (
                "Backpropagation is the algorithm used to train a neural network by efficiently "
                "computing the gradient of the loss function with respect to every weight in the "
                "network. Training has two phases. In the forward pass, input data flows through the "
                "layers, each neuron computes a weighted sum of its inputs plus a bias and applies an "
                "activation function such as ReLU or sigmoid, and the network produces a prediction "
                "whose error is measured by the loss function. In the backward pass, the error is "
                "propagated from the output layer back towards the input layer using the chain rule of "
                "calculus, so the gradient for each layer is computed from the gradient of the layer "
                "after it. These gradients are then used by an optimizer such as gradient descent to "
                "update the weights and biases in the direction that reduces the loss, and the process "
                "is repeated over many epochs until the loss converges. Deep networks can suffer from "
                "the vanishing gradient problem and the exploding gradient problem, which are reduced "
                "with ReLU activations, careful weight initialization, batch normalization, gradient "
                "clipping and residual connections."
            ),
            "keywords": [
                "neural network",
                "forward pass|forward propagation",
                "backward pass|backward propagation",
                "chain rule",
                "gradient",
                "loss function|cost function",
                "weights",
                "activation function",
                "vanishing gradient|exploding gradient",
            ],
        },
        {
            "id": "AIML-06",
            "topic": "Classification Metrics",
            "difficulty": "Intermediate",
            "question": (
                "What are precision, recall and F1-score, and why can accuracy be misleading?"
            ),
            "ideal_answer": (
                "Precision, recall and F1-score are classification evaluation metrics derived from the "
                "confusion matrix, which counts true positives, false positives, true negatives and "
                "false negatives. Precision is TP divided by TP plus FP: of all the instances the model "
                "predicted as positive, how many were truly positive, so high precision means few "
                "false alarms. Recall, also called sensitivity or the true positive rate, is TP divided "
                "by TP plus FN: of all the actual positives, how many the model correctly found, so "
                "high recall means few missed cases. The F1-score is the harmonic mean of precision "
                "and recall, two times precision times recall divided by precision plus recall, and "
                "gives a single balanced measure. Accuracy can be misleading on an imbalanced dataset: "
                "if only one percent of transactions are fraud, a model that always predicts not fraud "
                "scores ninety-nine percent accuracy while catching no fraud at all. Precision matters "
                "more when a false positive is costly, such as spam filtering, and recall matters more "
                "when a false negative is costly, such as cancer detection. There is usually a "
                "precision-recall tradeoff that is controlled by the classification threshold."
            ),
            "keywords": [
                "confusion matrix",
                "true positive",
                "false positive",
                "false negative",
                "precision",
                "recall|sensitivity",
                "harmonic mean",
                "imbalanced dataset|imbalanced data|class imbalance",
                "threshold",
            ],
        },
        {
            "id": "AIML-07",
            "topic": "TF-IDF & Cosine Similarity",
            "difficulty": "Advanced",
            "question": (
                "Explain TF-IDF and cosine similarity, and how they can be used to compare two "
                "pieces of text."
            ),
            "ideal_answer": (
                "TF-IDF, which stands for term frequency - inverse document frequency, is a technique "
                "for converting text into numerical vectors that reflect how important a word is to a "
                "document within a collection called a corpus. Term frequency measures how often a "
                "term appears in a document, while inverse document frequency is calculated as the "
                "logarithm of the total number of documents divided by the number of documents that "
                "contain the term, so very common words such as 'the' receive a low weight and rare, "
                "informative words receive a high weight. The TF-IDF weight is the product of TF and "
                "IDF, and each document becomes a sparse vector in a high-dimensional vector space. "
                "Cosine similarity then measures how similar two such vectors are by computing the "
                "cosine of the angle between them, which is the dot product of the vectors divided by "
                "the product of their magnitudes. For TF-IDF vectors the value ranges from 0, meaning "
                "no shared terms, to 1, meaning the same direction, and because it depends on "
                "direction rather than magnitude it is not biased by document length. Together they "
                "are widely used in information retrieval, search engines, document clustering, "
                "plagiarism detection and automated answer evaluation, although they capture word "
                "overlap rather than deep meaning, so synonyms are not recognized unless word "
                "embeddings are used."
            ),
            "keywords": [
                "term frequency",
                "inverse document frequency",
                "corpus",
                "vector",
                "logarithm|log",
                "cosine similarity",
                "dot product",
                "magnitude|norm",
                "angle",
            ],
        },
    ],
    # =========================================================================
    # DOMAIN 2: PYTHON & DATA SCIENCE
    # =========================================================================
    "Python & Data Science": [
        {
            "id": "PYDS-01",
            "topic": "Lists vs Tuples",
            "difficulty": "Foundational",
            "question": "What is the difference between a list and a tuple in Python? When would you use each?",
            "ideal_answer": (
                "Lists and tuples are both ordered sequence types in Python that can store "
                "heterogeneous elements and support indexing, slicing and iteration. The key "
                "difference is mutability: a list is mutable, so elements can be added, removed or "
                "changed in place with methods such as append, extend, insert, pop and remove, while a "
                "tuple is immutable and cannot be modified after it is created. Lists are written with "
                "square brackets and tuples with parentheses. Because tuples are immutable they are "
                "hashable when all their elements are hashable, so they can be used as dictionary keys "
                "or set elements, whereas lists cannot. Tuples have a smaller memory footprint and are "
                "slightly faster to create and iterate, which makes them suitable for fixed "
                "collections such as coordinates, RGB values, function return values and database "
                "records. Lists are preferred for dynamic collections whose size or content changes. "
                "Tuple immutability also makes code safer, because the data cannot be accidentally "
                "modified when it is passed between functions."
            ),
            "keywords": [
                "mutable",
                "immutable",
                "ordered",
                "square brackets",
                "parentheses|round brackets",
                "hashable",
                "dictionary keys|dict keys",
                "memory",
                "append",
            ],
        },
        {
            "id": "PYDS-02",
            "topic": "Decorators",
            "difficulty": "Intermediate",
            "question": "What are decorators in Python? Explain how they work with a use case.",
            "ideal_answer": (
                "A decorator in Python is a function that takes another function as an argument, adds "
                "extra behaviour to it, and returns a new function, without permanently modifying the "
                "original function's source code. This is possible because functions are first-class "
                "objects in Python: they can be passed as arguments, returned from other functions and "
                "assigned to variables. A decorator is usually implemented as a closure, where an inner "
                "wrapper function accepts *args and **kwargs, runs code before and after calling the "
                "original function, and returns its result. The @decorator syntax placed above a "
                "function definition is syntactic sugar for writing my_function = "
                "decorator(my_function). It is good practice to apply functools.wraps to the wrapper "
                "so that the original function's name and docstring are preserved. Decorators are "
                "widely used for logging, timing and performance measurement, caching with "
                "functools.lru_cache, authentication and access control in web frameworks such as "
                "Flask and Django, input validation and retry logic. Decorators can also take their "
                "own arguments by adding another level of nesting, and a class can act as a decorator "
                "by implementing the __call__ method."
            ),
            "keywords": [
                "first-class objects|first-class functions|first class",
                "wrapper function|wrapper",
                "closure",
                "kwargs|keyword arguments",
                "syntactic sugar|@ syntax",
                "functools.wraps|wraps",
                "logging",
                "caching|lru_cache",
            ],
        },
        {
            "id": "PYDS-03",
            "topic": "Generators",
            "difficulty": "Intermediate",
            "question": "What is a generator in Python, and how is yield different from return?",
            "ideal_answer": (
                "A generator in Python is a special kind of iterator that produces values lazily, one "
                "at a time, instead of building and storing an entire sequence in memory. It is "
                "defined like a normal function but uses the yield keyword instead of return. When a "
                "generator function is called it returns a generator object without running the body; "
                "each call to next() resumes execution from where it left off, runs until the next "
                "yield, and produces that value while preserving the function's local state between "
                "calls. When the function finishes, a StopIteration exception is raised, which a for "
                "loop handles automatically. The difference from return is that return terminates the "
                "function and sends back a single value, while yield pauses the function and can "
                "produce a series of values over time. This lazy evaluation makes generators memory "
                "efficient for large files, data streams and infinite sequences. Generator "
                "expressions, written with parentheses such as (x * x for x in range(10)), are a "
                "compact alternative to list comprehensions. Generators can also be chained together "
                "to build efficient data-processing pipelines."
            ),
            "keywords": [
                "iterator",
                "yield",
                "lazy evaluation|lazily",
                "memory efficient|memory",
                "next",
                "local state|state",
                "stopiteration",
                "generator expressions",
            ],
        },
        {
            "id": "PYDS-04",
            "topic": "Global Interpreter Lock",
            "difficulty": "Advanced",
            "question": (
                "What is the Global Interpreter Lock (GIL) in Python, and how does it affect "
                "multithreading versus multiprocessing?"
            ),
            "ideal_answer": (
                "The Global Interpreter Lock, or GIL, is a mutex in CPython, the standard Python "
                "interpreter, that allows only one thread to execute Python bytecode at a time, even "
                "on a multi-core processor. It exists mainly because CPython's memory management uses "
                "reference counting, which is not thread-safe, and a single global lock keeps the "
                "interpreter simple and fast for single-threaded code. As a result, multithreading "
                "does not give true parallelism for CPU-bound tasks such as heavy numerical "
                "computation, because threads take turns holding the GIL. However, threads are still "
                "very useful for I/O-bound tasks such as network requests, file operations and "
                "database calls, because the GIL is released while a thread waits for I/O. For "
                "CPU-bound work, the multiprocessing module or a ProcessPoolExecutor runs separate "
                "processes, each with its own interpreter and its own GIL, achieving real parallel "
                "execution across cores at the cost of higher memory use and inter-process "
                "communication. Libraries such as NumPy release the GIL inside optimized C code, and "
                "asyncio provides concurrency for I/O-bound workloads within a single thread. Recent "
                "Python versions also offer an optional free-threaded build that can run without the "
                "GIL."
            ),
            "keywords": [
                "global interpreter lock|gil",
                "cpython",
                "mutex|lock",
                "thread",
                "cpu-bound|cpu bound",
                "i/o-bound|io-bound|io bound",
                "multiprocessing",
                "reference counting",
                "parallelism|parallel",
            ],
        },
        {
            "id": "PYDS-05",
            "topic": "Missing Values in Pandas",
            "difficulty": "Foundational",
            "question": "How do you detect and handle missing values in a Pandas DataFrame?",
            "ideal_answer": (
                "Missing values in Pandas are represented as NaN, or None and NaT for datetimes, and "
                "must be handled before analysis or model training because most machine learning "
                "algorithms cannot process them. The first step is to detect them with isnull() or "
                "isna() combined with sum() to count missing values per column, and to understand "
                "why the data is missing, whether completely at random or following a pattern. One "
                "option is deletion with dropna(), which removes rows or columns that contain missing "
                "values and is suitable when only a small fraction of the data is missing; the thresh "
                "and subset parameters give finer control. The other option is imputation with "
                "fillna(): numerical columns are commonly filled with the mean, or the median when "
                "outliers are present, categorical columns with the mode, and time series data with "
                "forward fill or backward fill, or by interpolation using interpolate(). More advanced "
                "approaches include scikit-learn's SimpleImputer and KNNImputer, model-based "
                "imputation, or adding an indicator column that flags which values were missing. The "
                "imputation strategy should be fitted on the training data only to avoid data leakage."
            ),
            "keywords": [
                "nan|null",
                "isnull|isna",
                "dropna",
                "fillna",
                "imputation|impute",
                "mean",
                "median",
                "mode",
                "forward fill|ffill",
                "interpolation|interpolate",
            ],
        },
        {
            "id": "PYDS-06",
            "topic": "NumPy Vectorization & Broadcasting",
            "difficulty": "Intermediate",
            "question": "Explain vectorization and broadcasting in NumPy. Why are they faster than Python loops?",
            "ideal_answer": (
                "Vectorization in NumPy means performing operations on entire arrays at once instead "
                "of writing explicit Python for loops over individual elements. A NumPy ndarray stores "
                "elements of a single data type in contiguous memory, and vectorized operations are "
                "executed by optimized, precompiled C code, so they avoid the per-element overhead of "
                "the Python interpreter and are often ten to a hundred times faster than equivalent "
                "loops. They also make code shorter and more readable, for example a * b + c for "
                "element-wise arithmetic, or universal functions (ufuncs) such as np.sqrt and np.exp. "
                "Broadcasting is the set of rules that lets NumPy perform element-wise operations on "
                "arrays of different shapes without copying data. NumPy compares shapes starting from "
                "the trailing dimensions: two dimensions are compatible if they are equal or one of "
                "them is 1, and a dimension of size 1 is virtually stretched to match the other. For "
                "example, adding an array of shape (3, 1) to an array of shape (1, 4) produces a "
                "(3, 4) result, and subtracting a row of column means from a matrix centres every "
                "column. If the shapes are not compatible, NumPy raises a ValueError."
            ),
            "keywords": [
                "arrays|ndarray",
                "for loops|python loops",
                "contiguous memory",
                "c code|compiled",
                "element-wise",
                "universal functions|ufuncs",
                "broadcasting",
                "shapes",
                "trailing dimensions",
                "compatible",
            ],
        },
        {
            "id": "PYDS-07",
            "topic": "Feature Scaling",
            "difficulty": "Intermediate",
            "question": (
                "What is the difference between normalization and standardization, and when should "
                "each be used?"
            ),
            "ideal_answer": (
                "Feature scaling transforms numerical features to a common scale so that features with "
                "large ranges do not dominate those with small ranges. It is essential for "
                "distance-based algorithms such as k-nearest neighbors, k-means clustering and support "
                "vector machines, for gradient descent based models such as linear regression, "
                "logistic regression and neural networks, which converge faster on scaled data, and "
                "for PCA. Normalization, usually min-max scaling, rescales each feature to a fixed "
                "range, typically 0 to 1, using x minus min divided by max minus min; it preserves the "
                "shape of the original distribution but is sensitive to outliers. Standardization, or "
                "z-score scaling, transforms each feature to have a mean of 0 and a standard deviation "
                "of 1 using x minus the mean divided by the standard deviation; it is less affected by "
                "outliers and suits data that is roughly Gaussian or models that assume centred data. "
                "Tree-based models such as decision trees and random forests generally do not need "
                "scaling. In scikit-learn these are implemented by MinMaxScaler and StandardScaler, and "
                "the scaler must be fitted only on the training set and then applied to the test set "
                "to avoid data leakage."
            ),
            "keywords": [
                "feature scaling|scaling",
                "normalization|min-max scaling|minmaxscaler",
                "standardization|z-score|standardscaler",
                "mean",
                "standard deviation",
                "range|0 to 1",
                "outliers",
                "distance-based|distance based|k-nearest neighbors|knn",
                "data leakage|leakage",
            ],
        },
    ],
}

REQUIRED_FIELDS = ("id", "question", "ideal_answer", "keywords")


# =============================================================================
# ACCESSOR HELPERS
# =============================================================================

def get_domains() -> List[str]:
    """Return the list of subject domains in display order."""
    return list(QUESTION_BANK.keys())


def get_questions(domain: str) -> List[Dict[str, object]]:
    """Return all question entries for ``domain`` (raises ``KeyError`` if unknown)."""
    if domain not in QUESTION_BANK:
        raise KeyError(f"Unknown domain: {domain!r}. Available: {get_domains()}")
    return QUESTION_BANK[domain]


def iter_all_questions() -> Iterator[Dict[str, object]]:
    """Yield every question entry across all domains."""
    for questions in QUESTION_BANK.values():
        yield from questions


def get_question(question_id: str) -> Dict[str, object]:
    """Look up a single question entry by its ``id``."""
    for entry in iter_all_questions():
        if entry["id"] == question_id:
            return entry
    raise KeyError(f"Unknown question id: {question_id!r}")


def get_domain_of(question_id: str) -> str:
    """Return the domain that contains ``question_id``."""
    for domain, questions in QUESTION_BANK.items():
        if any(q["id"] == question_id for q in questions):
            return domain
    raise KeyError(f"Unknown question id: {question_id!r}")


def get_reference_corpus() -> List[str]:
    """All model answers - used to fit domain-aware IDF statistics."""
    return [str(entry["ideal_answer"]) for entry in iter_all_questions()]


def keyword_display_name(keyword: str) -> str:
    """Human-readable label for a keyword spec (first ``|`` alternative)."""
    return keyword.split("|")[0].strip()


def validate_question_bank() -> None:
    """Raise ``ValueError`` if the bank is malformed (missing fields, duplicate IDs...)."""
    seen_ids = set()
    for domain, questions in QUESTION_BANK.items():
        if not questions:
            raise ValueError(f"Domain {domain!r} has no questions.")
        for entry in questions:
            missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
            if missing:
                raise ValueError(f"Question {entry.get('id', '?')} in {domain!r} is missing {missing}.")
            qid = str(entry["id"])
            if qid in seen_ids:
                raise ValueError(f"Duplicate question id: {qid}")
            seen_ids.add(qid)
            keywords = entry["keywords"]
            if not isinstance(keywords, list) or not all(isinstance(k, str) and k.strip() for k in keywords):
                raise ValueError(f"Question {qid}: 'keywords' must be a list of non-empty strings.")
