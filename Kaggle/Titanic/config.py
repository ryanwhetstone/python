values = {
    'index_column': "PassengerId",
    'target_column': "Survived",
    'fold_type': 'stratified',  # None, stratified, kfold
    'n_fold': 7,
	'debug': False,
	'debug_without_one_hot': False,
	'use_transformation': 'binary', # all or specific name of transformation (binary/continuous/normalize)
	'show_model_optimizations': True, # True/False

    'original_data_path': "data/original/",
    'output_data_path': "data/output/",
    'transformed_data_path': "data/transformed/",
    'models_data_path': "data/models/",
    'graphs_data_path': "data/graphs/",
    'test_file': "test.csv",
    'train_file': "train.csv",

    'main_transformations': {
		'binary': [
			# Options for columns that are missing data either in the training or test data sets
			['Cabin', ['nan_static', '_']],
			['Embarked', ['nan_static', 'S']],

			['Title', ['copy_column', 'Name']],
			["Title", ["regex_extract", " ([A-Za-z]+)\."]],
			["Title", ["str_replace", ['Lady', 'Countess', 'Dona'], 'Royalty']],
			["Title", ["str_replace", ['Capt', 'Col', 'Major', 'Rev'], 'Officer']],
			["Title", ["str_replace", ['Jonkheer', 'Don', 'Sir'], 'Royalty']],
			["Title", ["str_replace", ['Mlle', 'Ms'], 'Miss']],
			["Title", ["str_replace", 'Mme', 'Mrs']],

			['Fare', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],			
			['Age', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],
			
			['Title_numeric_categories', ['binary_numeric_categories', 'Title']],
			
			["FamilySize", ["add_columns", "Parch", "SibSp"]],
			# ['FamilySize', ['one_hot']],
			["IsAlone", ["extract_dummy", "FamilySize", '==', 0]],
			["SmFamily", ["extract_dummy", "FamilySize", '><', 1,2]],
			["MdFamily", ["extract_dummy", "FamilySize", '><', 3,4]],
			["LgFamily", ["extract_dummy", "FamilySize", '>=', 5]],


			# Transformations on features with string data
			['Pclass_one_hot', ['binary_one_hot', 'Pclass']],
			['Name_string_len', ['copy_column', 'Name']],
			['Name_string_len', ['string_length']],
			['Name_string_len_split_by_cut', ['binary_split_by_cut', 'Name_string_len', 5]],

			# ['Name_first_character', ['binary_first_character', 'Name']],
			# ['Sex_numeric_categories', ['binary_numeric_categories', 'Sex']],
			['Sex', ['numeric_categories']],
			# ['Age_split_qcut_6', ['binary_split_by_qcut', 'Age', 6]],
			["Age_split_defined", ["binary_split_by_defined", 'Age', [0, 11, 18, 22, 27, 33, 40, 66, 90]]],

			# ['Age_split_cut_4', ['binary_split_by_cut', 'Age', 4]],
			# ['Age_split_qcut_8', ['binary_split_by_qcut', 'Age', 8]],
			# ['Age_split_cut_8', ['binary_split_by_cut', 'Age', 8]],
			# ['SibSp_one_hot', ['binary_one_hot', 'SibSp']],
			# ['Parch_one_hot', ['binary_one_hot', 'Parch']],
			# ['Ticket_string_len', ['binary_string_length', 'Ticket']],
			# ['Ticket_first_character', ['binary_first_character', 'Ticket']],
			['Fare_split_cut_5', ['binary_split_by_qcut', 'Fare', 5]],
			# ['Fare_split_cut_4', ['binary_split_by_cut', 'Fare', 4]],
			# ['Fare_split_qcut_8', ['binary_split_by_qcut', 'Fare', 8]],
			# ['Fare_split_cut_8', ['binary_split_by_cut', 'Fare', 8]],
			# ['Cabin_string_len', ['binary_string_length', 'Cabin']],
			['Cabin_first_character', ['binary_first_character', 'Cabin']],
			['Embarked_numeric_categories', ['binary_numeric_categories', 'Embarked']],

			# Drop statements for all columns, uncomment any you want to drop
			# ['PassengerId', ['drop_column']],
			['FamilySize', ['drop_column']],
			['Name_string_len', ['drop_column']],
			['Title', ['drop_column']],
			['Pclass', ['drop_column']],
			['Name', ['drop_column']],
			['Sex', ['drop_column']],
			['Age', ['drop_column']],
			['SibSp', ['drop_column']],
			['Parch', ['drop_column']],
			['Ticket', ['drop_column']],
			['Fare', ['drop_column']],
			['Cabin', ['drop_column']],
			['Embarked', ['drop_column']],

			["Cabin_first_character_8",             ["drop_column"]],

		],
		'continuous': [
			# Options for columns that are missing data either in the training or test data sets
			['Cabin', ['nan_static', '_']],
			["Cabin", ["first_character"]],

			['Embarked', ['nan_static', 'S']],

			['Title', ['copy_column', 'Name']],
			["Title", ["regex_extract", " ([A-Za-z]+)\."]],
			["Title", ["str_replace", ['Lady', 'Countess', 'Dona'], 'Royalty']],
			["Title", ["str_replace", ['Capt', 'Col', 'Major', 'Rev'], 'Officer']],
			["Title", ["str_replace", ['Jonkheer', 'Don', 'Sir'], 'Royalty']],
			["Title", ["str_replace", ['Mlle', 'Ms'], 'Miss']],
			["Title", ["str_replace", 'Mme', 'Mrs']],
			['Age', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],

			['Fare', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],
			['Title', ['numeric_categories']],
			
			
			# Transformations on features with string data
			['Name_string_len', ['continuous_string_length', 'Name']],
			['Name_first_character', ['continuous_first_character', 'Name']],
			['Sex_numeric_categories', ['continuous_numeric_categories', 'Sex']],
			['Ticket_string_len', ['continuous_string_length', 'Ticket']],
			['Ticket_first_character', ['continuous_first_character', 'Ticket']],
			['Cabin_string_len', ['continuous_string_length', 'Cabin']],
			['Cabin_first_character', ['continuous_first_character', 'Cabin']],
			['Embarked_numeric_categories', ['continuous_numeric_categories', 'Embarked']],

			# Drop statements for all columns, uncomment any you want to drop
			# ['PassengerId', ['drop_column']],
			['Pclass', ['drop_column']],
			['Name', ['drop_column']],
			['Sex', ['drop_column']],
			# ['Age', ['drop_column']],
			['SibSp', ['drop_column']],
			['Parch', ['drop_column']],
			['Ticket', ['drop_column']],
			# ['Fare', ['drop_column']],
			['Cabin', ['drop_column']],
			['Embarked', ['drop_column']],
		],
		'normalized': [
			# Options for columns that are missing data either in the training or test data sets
			['Cabin', ['nan_static', '_']],
			["Cabin", ["first_character"]],

			['Embarked', ['nan_static', 'S']],

			['Title', ['copy_column', 'Name']],
			["Title", ["regex_extract", " ([A-Za-z]+)\."]],
			["Title", ["str_replace", ['Lady', 'Countess', 'Dona'], 'Royalty']],
			["Title", ["str_replace", ['Capt', 'Col', 'Major', 'Rev'], 'Officer']],
			["Title", ["str_replace", ['Jonkheer', 'Don', 'Sir'], 'Royalty']],
			["Title", ["str_replace", ['Mlle', 'Ms'], 'Miss']],
			["Title", ["str_replace", 'Mme', 'Mrs']],
			['Age', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],

			['Fare', ['nan_mean_from_columns', ['Sex', 'Pclass', 'Title']]],
			['Title', ['numeric_categories']],
			['Title', ['normalize']],
			
			
			# Transformations on features with string data
			['Name_string_len', ['normalized_string_length', 'Name']],
			['Name_first_character', ['normalized_first_character', 'Name']],
			['Sex_numeric_categories', ['normalized_numeric_categories', 'Sex']],
			['Age_normalized', ['normalized_column', 'Age']],
			['SibSp_normalized', ['normalized_column', 'SibSp']],
			['Parch_normalized', ['normalized_column', 'Parch']],
			['Ticket_string_len', ['normalized_string_length', 'Ticket']],
			['Ticket_first_character', ['normalized_first_character', 'Ticket']],
			['Fare_normalized', ['normalized_column', 'Fare']],
			['Cabin_string_len', ['normalized_string_length', 'Cabin']],
			['Cabin_first_character', ['normalized_first_character', 'Cabin']],
			['Embarked_numeric_categories', ['normalized_numeric_categories', 'Embarked']],

			# Drop statements for all columns, uncomment any you want to drop
			# ['PassengerId', ['drop_column']],
			['Pclass', ['drop_column']],
			['Name', ['drop_column']],
			['Sex', ['drop_column']],
			['Age', ['drop_column']],
			['SibSp', ['drop_column']],
			['Parch', ['drop_column']],
			['Ticket', ['drop_column']],
			['Fare', ['drop_column']],
			['Cabin', ['drop_column']],
			['Embarked', ['drop_column']],
		]
	}
}
