from notebooks.heart_transplant.dependencies.heart_transplant_data import get_reduced_dataset

X, y, dataset_raw = get_reduced_dataset(survival_days=365)

X.assign(tx_year=dataset_raw['tx_year']).assign(y=y).to_csv('./data/heart_transplant/input_data_with_year.csv')
