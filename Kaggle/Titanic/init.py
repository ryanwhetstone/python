import os
import lib.functions as f
import config

os.makedirs(config.values['transformed_data_path'], exist_ok=True)
os.makedirs(config.values['original_data_path'], exist_ok=True)
os.makedirs(config.values['models_data_path'], exist_ok=True)
os.makedirs(config.values['graphs_data_path'], exist_ok=True)
os.makedirs(config.values['output_data_path'], exist_ok=True)
os.makedirs(config.values['output_data_path'] + 'test/', exist_ok=True)
os.makedirs(config.values['output_data_path'] + 'train/', exist_ok=True)

f.remove_files(config.values['transformed_data_path'])
f.remove_files(config.values['models_data_path'])
f.remove_files(config.values['graphs_data_path'])
f.remove_files(config.values['output_data_path'] + 'test')
f.remove_files(config.values['output_data_path'] + 'train')
f.remove_files(config.values['output_data_path'])

