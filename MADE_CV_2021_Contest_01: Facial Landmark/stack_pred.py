import pandas as pd
from tqdm import tqdm

submit_list = ["./runs/resnet152n0_noise_submit.csv",
               './runs/resnet18no_grad_submit.csv',
               './runs/resnet50no_grad_submit.csv',
               './runs/resnet101no_grad_submit.csv',
               './runs/resnext101_32x8d_no_noise_submit.csv',
               './runs/resnet1011sh_submit.csv',
               './runs/resnet101sh_submit.csv',
               './runs/resnet18aug_submit.csv',
               './runs/resnet18sh_submit.csv',
               './runs/resnet50sh_submit.csv'
               ]

dfs_list = []

for path in submit_list:
    dfs_list.append(pd.read_csv(path))

columns = dfs_list[0].columns[1:]

for col_name in tqdm(columns):
    a = dfs_list[0][col_name]
    for df in dfs_list[1:]:
        a += df[col_name]
    a /= len(dfs_list)

    dfs_list[0][col_name] = round(a).astype('int32')

dfs_list[0].to_csv('stack.csv', index=False)

print('vse')
