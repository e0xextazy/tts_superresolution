import pandas as pd

orig_count = 0
target_count = 0
pred_count = 0
input_count = 0

orig_sum = 0
target_sum = 0
pred_sum = 0
input_sum = 0

# df = pd.read_csv('mos.tsv', sep='\t')
all_table = pd.read_table('dynamic_unet/mos.tsv')
col_names = ['INPUT:audio', 'OUTPUT:opinion_score']
table = all_table[col_names]
table = table.dropna()

for index, row in table.iterrows():
    if 'orig' in row['INPUT:audio']:
        orig_count += 1
        orig_sum += row['OUTPUT:opinion_score']
    elif 'target' in row['INPUT:audio']:
        target_count += 1
        target_sum += row['OUTPUT:opinion_score']
    elif 'pred' in row['INPUT:audio']:
        pred_count += 1
        pred_sum += row['OUTPUT:opinion_score']
    else:
        input_count += 1
        input_sum += row['OUTPUT:opinion_score']

print('orig mos: ', orig_sum/orig_count, 'orig count:', orig_count)
print('target mos: ', target_sum/target_count, 'target count:', target_count)
print('pred mos: ', pred_sum/pred_count, 'pred count:', pred_count)
print('input mos: ', input_sum/input_count, 'input count:', input_count)
