import logging
import numpy as np
import pandas
from functools import partial
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from toolz import identity
from typing import Callable, Tuple, List

from cache import memory
from functional import pipe, statements, tap
from notebooks.heart_transplant.dependencies.heart_transplant_functions import log_transform_dataset, \
    filter_out_unused_features, get_X_y_1_year_survival, SURVIVAL_DAYS_DEFAULT, get_filtered_by_age, \
    remove_missing_columns, get_rolling_cv, AgeGroup
from notebooks.heart_transplant.dependencies.heart_transplant_metadata import heart_transplant_metadata as metadata
from visualisation import format_number


def type_conversion(dataset):
    dataset_new = dataset.copy()
    dataset_new['height ratio'] = pandas.to_numeric(dataset['height ratio'])
    dataset_new['weight ratio'] = pandas.to_numeric(dataset['weight ratio'])
    dataset_new['tx_year'] = pandas.to_numeric(dataset['tx_year'])
    return dataset_new


def convert_missing_codes_to_na(_X):
    _X_new = _X.copy()
    for column in _X.columns:
        try:
            metadata_record_na = metadata[column]['na_values']
        except KeyError:
            pass
        else:
            _X_new[column].replace(metadata_record_na, np.nan, inplace=True)
    return _X_new


def remove_last_year(_X):
    return _X[_X['tx_year'] <= 2016]


def keep_only_heart(_X: DataFrame) -> DataFrame:
    return _X[_X['organ'] == 'HR']


def get_base_dataset(log_transform: bool = True):
    dataset_raw = pandas.read_csv("../cardiovascular-risk-data/data/UNOS_v3.csv")
    dataset_raw.columns = [column.lower() for column in dataset_raw.columns]
    return (
        pipe(
            dataset_raw,
            tap(lambda d: logging.info(f'Raw dataset, n= {format_number(len(d))}')),
            convert_missing_codes_to_na,
            type_conversion,
            keep_only_heart,
            tap(lambda d: logging.info(f'Only heart, n= {format_number(len(d))}')),
            remove_last_year,
            tap(lambda d: logging.info(f'Last year removed, n= {format_number(len(d))}')),
            partial(filter_out_unused_features, metadata=metadata),
            tap(lambda d: logging.info(f'Input features {format_number(len(d.columns))}')),
            log_transform_dataset if log_transform else identity,
        ),
        dataset_raw,
    )


def get_survival_dataset(survival_days: int = 365):
    X, y, dataset_raw = get_reduced_binary_dataset(survival_days=survival_days, log_transform=True)
    return X, y, dataset_raw['futd'].loc[X.index], dataset_raw['death'].loc[X.index], dataset_raw


get_survival_dataset_cached = memory.cache(get_survival_dataset)


def get_binary_dataset(survival_days: int = SURVIVAL_DAYS_DEFAULT, log_transform: bool = True):
    dataset, dataset_raw = get_base_dataset(log_transform=log_transform)

    X, y = pipe(
        dataset,
        partial(get_X_y_1_year_survival, survival_days=survival_days),
    )

    return X, y, dataset_raw


def get_reduced_binary_dataset(
    survival_days: int = SURVIVAL_DAYS_DEFAULT, log_transform: bool = True
):
    X, y, dataset_raw = get_binary_dataset(survival_days=survival_days, log_transform=log_transform)

    X_reduced, y_reduced = reduce_dataset(X, y)

    logging.info(f'Dropping rows with too many missing variables: {len(X)} -> {len(X_reduced)}')

    return X_reduced, y_reduced, dataset_raw


def reduce_dataset(X, y):
    missing_mask = X.copy().swifter.apply(lambda x: x.count() >= 80, axis=1)
    X_reduced = X[missing_mask]
    y_reduced = y[missing_mask]
    return X_reduced, y_reduced


get_reduced_binary_dataset_cached = memory.cache(get_reduced_binary_dataset)
get_binary_dataset_cached = memory.cache(get_binary_dataset)

__NOT_USED__EXPERTISE_BASED_FEATURES = [
    'ischtime', 'tbili', 'creat_trr', 'cmassratio', 'hist_mi', 'congenital', 'ecd_donor',
    'wgt_kg_calc', 'sgpt_don', 'sgot_don', 'age_don', 'age', 'cdc_risk_hiv_don', 'bun_don',
    'tbili_don', 'biopsy_dgn', 'creat_don', 'vessels_50sten', 'most_rcnt_creat', 'lv_eject',
    'education', 'retransplant', 'newpra', 'protein_urine', 'hematocrit_don', 'iabp_tcr',
    'hiv_serostatus', 'gender', 'cmv_status', 'hemo_co_tcr', 'impl_defibril', 'hist_hypertens_don',
    'malig', 'tot_serum_album', 'dial_after_list', 'hist_cig_don', 'ethcat', 'tah ever',
    'prior_card_surg_type_tcr', 'med_cond_trr', 'rvad ever', 'cig_use', 'donor insulin', 'ecmo_trr',
    'gender_don', 'ebv_serostatus', 'hbv_core_don', 'diabetes_don', 'thoracic_dgn', 'multiorg',
    'hist_alcohol_old_don', 'hist_cancer_don', 'cmv_don', 'cod_cad_don', 'lvad ever',
    'hemo_sys_tcr', 'cereb_vasc', 'other_inf_don', 'dial_prior_tx', 'abo_don', 'vasodil_don',
    'bmi_calc', 'diab', 'skin_cancer_don', 'hemo_pa_dia_tcr', 'hbv_sur_antigen_don',
    'inotropes_tcr', 'hist_cocaine_don', 'alcohol_heavy_don', 'abo', 'ventilator_tcr',
    'hbv_sur_antigen', 'prior_card_surg_trr', 'pulm_inf_don', 'hcv_serostatus',
    'vent_support_after_list', 'hbv_core', 'intracranial_cancer_don', 'transfusions', 'prev_tx',
    'contin_cocaine_don', 'extracranial_cancer_don', 'hep_c_anti_don', 'vdrl_don',
    'clin_infect_don', 'blood_inf_don', 'diag', 'hemo_pcw_tcr'
]


def get_base_inputs(
    get_sampling_sets,
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
    filter_callback: Callable = identity,
) -> Tuple[DataFrame, Series, DataFrame, DataFrame, List[List]]:
    X, y, dataset_raw = get_reduced_binary_dataset_cached(survival_days)

    logging.info(f'Loaded n={len(X)}')

    X_filtered, y_filtered = get_filtered_by_age(group, X, y)
    X_filtered = filter_callback(X_filtered)
    y_filtered = y.loc[X_filtered.index]

    logging.info(f'Filtered, n={len(X_filtered)}')

    sampling_sets = get_sampling_sets(X_filtered, y_filtered, dataset_raw)

    logging.info(f'Cross-validation folds n={sampling_sets}')

    X_valid = remove_missing_columns(X_filtered, sampling_sets, verbose=1)

    logging.info(
        f'Removing features with 0 variance in some fold: {len(X_filtered.columns)} ➡ {len(X_valid.columns)}'
    )

    return X_filtered, y_filtered, X_valid, dataset_raw, sampling_sets


def get_rolling_cv_inputs(
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
) -> Tuple[DataFrame, Series, DataFrame, DataFrame, List[List]]:
    return get_base_inputs(
        lambda X, y, dataset_raw: list(
            get_rolling_cv(
                X.assign(tx_year=dataset_raw['tx_year']),
                n_windows=None,
                test_size_years=1,
                minimum_training_years=10,
                year_stop=2016
            )
        ),
        survival_days=survival_days,
        group=group,
    )


get_rolling_cv_inputs_cached = memory.cache(get_rolling_cv_inputs)


def get_rolling_cv_inputs_for_test(
    group: AgeGroup = AgeGroup.ALL, survival_days: int = SURVIVAL_DAYS_DEFAULT
):
    return get_base_inputs(
        lambda X, y, dataset_raw: list(
            get_rolling_cv(
                X.assign(tx_year=dataset_raw['tx_year']),
                n_windows=None,
                test_size_years=1,
                minimum_training_years=10,
                year_stop=2016
            )
        ),
        survival_days=survival_days,
        group=group,
        filter_callback=lambda X: statements(
            X_subsampled := X.sample(10000),
            X_subsampled := X_subsampled.iloc[:, :10],
            X_subsampled,
        )
    )


get_expanding_window_inputs_for_test_cached = memory.cache(get_rolling_cv_inputs_for_test)


def get_shuffled_cv_inputs():
    return get_base_inputs(
        lambda X, y, dataset_raw: list(KFold(n_splits=10, shuffle=True).split(X, y))
    )


get_shuffled_cv_inputs_cached = memory.cache(get_shuffled_cv_inputs)
