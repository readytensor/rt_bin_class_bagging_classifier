import json
import pandas as pd

from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from prediction.predictor_model import (
    evaluate_predictor_model,
    save_predictor_model,
    train_predictor_model,
)
from preprocessing.preprocess import (
    handle_class_imbalance,
    insert_nulls_in_nullable_features,
    save_pipeline_and_target_encoder,
    train_pipeline_and_target_encoder,
    transform_data,
)
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
from xai.explainer import fit_and_save_explainer

logger = get_logger(task_name="train")


def tune_k_for_smote(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    valid_X: pd.DataFrame,
    valid_y: pd.Series,
    hyperparameters: dict,
    hpt_specs_file_path: str,
    hpt_results_dir_path: str,
) -> int:
    """
    Tune the nearest neighbors parameter for the SMOTE algorithm.

    Args:
        train_X: train inputs.
        train_y: train labels.
        valid_X: valid inputs.
        valid_y: valid labels.
        hyperparameters: The hyperparameters to use for training the model.
        hpt_specs_file_path: Path to the hyperparameter tuning specifications file.
        hpt_results_dir_path: Directory to save hyperparameter tuning results.

    Returns:
        int: The best k found.
    """
    logger.info("Tuning K for Smote...")
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    grid_search_vals = hpt_specs["k_neighbors"]["grid_search_vals"]
    best_val_score = -1
    best_k = None
    hpt_vals = []
    for k_val in grid_search_vals:

        balanced_train_inputs, balanced_train_targets = handle_class_imbalance(
            train_X.copy(), train_y.copy(), k_neighbors=k_val
        )
        predictor = train_predictor_model(
            balanced_train_inputs,
            balanced_train_targets,
            hyperparameters,
        )
        val_score = evaluate_predictor_model(predictor, valid_X, valid_y)
        logger.info(f"k_val = {k_val}, val_score = {val_score}")
        hpt_vals.append({"k_val": k_val, "val_score": val_score})
        if val_score > best_val_score:
            best_val_score = val_score
            best_k = k_val

    logger.info(f"Best k_val = {best_k}; " f"Validation score = {best_val_score}")

    # save hyperparameter tuning results
    logger.info("Saving hyperparameter tuning results...")
    hpt_results_file_path = f"{hpt_results_dir_path}/smote_k.json"
    with open(hpt_results_file_path, "w", encoding="utf-8") as f_h:
        json.dump(hpt_vals, f_h, indent=4)

    return best_k


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    preprocessing_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
    hpt_results_dir_path: str = paths.HPT_OUTPUTS_DIR,
    explainer_config_file_path: str = paths.EXPLAINER_CONFIG_FILE_PATH,
    explainer_dir_path: str = paths.EXPLAINER_DIR_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model
            configuration file.
        train_dir (str, optional): The directory path of the train data.
        preprocessing_config_file_path (str, optional): The path of the preprocessing
            configuration file.
        preprocessing_dir_path (str, optional): The dir path where to save the pipeline
            and target encoder.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default
            hyperparameters file.
        run_tuning (bool, optional): Whether to run hyperparameter tuning.
            Default is False.
        hpt_specs_file_path (str, optional): The path of the configuration file for
            hyperparameter tuning.
        hpt_results_dir_path (str, optional): Dir path where to save the HPT results.
        explainer_config_file_path (str, optional): The path of the explainer
            configuration file.
        explainer_dir_path (str, optional): Dir path where to save the explainer.
    Returns:
        None
    """
    try:

        logger.info("Starting training...")
        # load and save schema
        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        # load model config
        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        # set seeds
        logger.info("Setting seeds...")
        set_seeds(seed_value=model_config["seed_value"])

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)

        # validate the data
        logger.info("Validating train data...")
        validated_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )

        # split train data into training and validation sets
        logger.info("Performing train/validation split...")
        train_split, val_split = split_train_val(
            validated_data, val_pct=model_config["validation_split"]
        )

        logger.info("Loading preprocessing config...")
        preprocessing_config = read_json_as_dict(preprocessing_config_file_path)

        # insert nulls in nullable features if no nulls exist in train data
        logger.info("Inserting nulls in nullable features if not present...")
        train_split_with_nulls = insert_nulls_in_nullable_features(
            train_split, data_schema, preprocessing_config
        )

        # fit and transform using pipeline and target encoder, then save them
        logger.info("Training preprocessing pipeline and label encoder...")
        pipeline, target_encoder = train_pipeline_and_target_encoder(
            data_schema, train_split_with_nulls, preprocessing_config
        )
        transformed_train_inputs, transformed_train_targets = transform_data(
            pipeline, target_encoder, train_split_with_nulls
        )
        transformed_val_inputs, transformed_val_targets = transform_data(
            pipeline, target_encoder, val_split
        )
        logger.info("Saving pipeline and label encoder...")
        save_pipeline_and_target_encoder(
            pipeline, target_encoder, preprocessing_dir_path
        )

        # tune k for smote
        hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
        best_k = tune_k_for_smote(
            transformed_train_inputs,
            transformed_train_targets,
            transformed_val_inputs,
            transformed_val_targets,
            hyperparameters,
            hpt_specs_file_path,
            hpt_results_dir_path,
        )

        logger.info("Handling class imbalance with Smote...")
        balanced_train_inputs, balanced_train_targets = handle_class_imbalance(
            transformed_train_inputs, transformed_train_targets, k_neighbors=best_k
        )

        # train model with default hyperparams
        logger.info("Training classifier...")
        hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
        predictor = train_predictor_model(
            balanced_train_inputs,
            balanced_train_targets,
            hyperparameters,
        )

        # save predictor model
        logger.info("Saving classifier...")
        save_predictor_model(predictor, predictor_dir_path)

        # calculate and print validation accuracy
        logger.info("Calculating score on validation data...")
        val_accuracy = evaluate_predictor_model(
            predictor, transformed_val_inputs, transformed_val_targets
        )
        logger.info(f"Validation score: {val_accuracy}")

        # fit and save explainer
        logger.info("Fitting and saving explainer...")
        _ = fit_and_save_explainer(
            transformed_train_inputs, explainer_config_file_path, explainer_dir_path
        )

        logger.info("Training completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
