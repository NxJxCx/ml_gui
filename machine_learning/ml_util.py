from typing import Any, Dict, Optional

from .base import MLBase
from .decision_tree import DecisionTreeClassification, DecisionTreeRegression
from .util import encode_base64, generate_session_id, map_list_json_compatible

ml_instances: Dict[str, MLBase] = {}


def get_all_algo_names() -> Dict[str, Dict[str, str]]:
    return {
        "Decision Tree": {
            DecisionTreeClassification._algorithm: DecisionTreeClassification._algo_name,
            DecisionTreeRegression._algorithm: DecisionTreeRegression._algo_name,
        },
    }


def get_class_from_algo(algo: str) -> Optional[type[MLBase]]:
    if algo == DecisionTreeClassification._algorithm:
        return DecisionTreeClassification
    elif algo == DecisionTreeRegression._algorithm:
        return DecisionTreeRegression
    return None


def set_session_id(session_instance, session_id: str):
    global ml_instances
    if not session_id:
        session_id = generate_session_id()
        session_instance["session_id"] = session_id
    if session_id not in ml_instances.keys():
        ml_instances[session_id] = None
    return session_id


def get_ml_instance(key: str):
    global ml_instances
    if key not in ml_instances.keys():
        return None
    return ml_instances[key]


def remove_ml_instance(key: str):
    global ml_instances
    if key in ml_instances.keys():
        ml_instances[key] = None


def clear_session_id(session_id: str):
    global ml_instances
    if session_id in ml_instances.keys():
        del ml_instances[session_id]


def set_ml_instance(key: str, algo: str, *args, **kwargs):
    global ml_instances
    if key in ml_instances.keys():
        AlgorithmClass = get_class_from_algo(algo)
        if AlgorithmClass:
            ml_instances[key] = AlgorithmClass(*args, **kwargs)


def get_plots(session_id: str) -> Dict[str, str]:
    global ml_instances
    if session_id in ml_instances.keys():
        ml = ml_instances.get(session_id, None)
        if not ml:
            return {}
        if isinstance(ml, DecisionTreeClassification):
            cd = encode_base64(ml.plot_class_distribution())
            cm = encode_base64(ml.plot_confusion_matrix())
            lc = encode_base64(ml.plot_learning_curve())
            dt = encode_base64(ml.plot_decision_tree())
            fi = encode_base64(ml.plot_feature_importance())
            return {
                "learning_curve": lc,
                "class_distribution": cd,
                "confusion_matrix": cm,
                "decision_tree": dt,
                "feature_importance": fi,
            }
        if isinstance(ml, DecisionTreeRegression):
            lc = encode_base64(ml.plot_learning_curve())
            rt = encode_base64(ml.plot_regression_tree())
            fi = encode_base64(ml.plot_feature_importance())
            return {
                "learning_curve": lc,
                "regression_tree": rt,
                "feature_importance": fi,
            }
    return {}


def get_algo_name(session_id: str) -> Optional[str]:
    global ml_instances
    if session_id not in ml_instances.keys():
        return None
    return ml_instances[session_id].algorithm


def get_trained_history_results(session_id: str) -> Dict[str, Any]:
    global ml_instances
    if session_id not in ml_instances.keys():
        return {}
    ml = ml_instances.get(session_id, None)
    if not ml:
        return {}
    evaluated = ml.evaluate_trained_model()
    indices = {i: str(k) for i, k in enumerate(evaluated.keys())}
    evaluation = map_list_json_compatible([evaluated[f"{indices[i]}"] for i in indices.keys()])
    evaluation = {f"{indices[i]}": evaluation[int(i)] for i in indices.keys()}
    plots = get_plots(session_id)
    algorithm = ml.algorithm
    algorithm_name = ml.algorithm_key
    dataset_dict = ml.data.to_dict()
    dataset = {k: list(map(lambda x: x[1], v.items())) for k, v in dataset_dict.items()}
    return {
        "plots": plots,
        "evaluation": evaluation,
        "algorithm": algorithm,
        "hyperparameters": ml.hyperparameters,
        "dataset": dataset,
        "algorithm_name": algorithm_name,
        "features": ml._features,
        "encoded_features": ml._encoded_features,
        "target": ml._target,
    }
