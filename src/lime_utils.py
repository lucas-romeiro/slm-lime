import re
import numpy as np
from IPython.display import display, HTML
from lime.lime_tabular import LimeTabularExplainer

def show_html(exp):
    display(HTML(exp.as_html()))

def extract_feature_name(rule):
    pattern = r"[a-zA-Z][a-zA-Z\s_]+"
    return re.search(pattern, rule).group().strip()

def explanation_to_dict(explanation):
    probs = explanation.predict_proba
    prediction_idx = int(np.argmax(probs))
    
    features_weights = explanation.as_list()
    
    structured_data = {
        "prediction": prediction_idx,
        "prediction_label": "Malignant" if prediction_idx == 1 else "Benign",
        "confidence": float(probs[prediction_idx]),
        "all_probabilities": {
            "benign": float(probs[0]),
            "malignant": float(probs[1])
        },
        "features": [extract_feature_name(f[0]) for f in features_weights],
        "weights": [float(f[1]) for f in features_weights],
        "feature_importance_map": {extract_feature_name(f[0]): float(f[1]) for f in features_weights}
    }
    
    return structured_data


def explanation_similarity(exp1, exp2, treshould=0.8):
    jacc = jaccard_similarity(exp1.keys(), exp2.keys())

    if jacc < treshould:
        return 1

    keys = set(exp1.keys()).intersection(exp2.keys())

    diff = []

    for k in keys:
        diff.append(abs(exp1[k] - exp2[k]))

    return np.mean(diff)

def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union
