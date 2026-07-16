#读取/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e_50/Rosetta_longbench_generate_20260706_132813_summary.json和/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e/Rosetta_longbench_generate_20260706_130600_summary.json文件，整理在一个表格中，每一行表示一个subject，每一列表示一个模型的结果，表格的第一列是subject名称，第二列是0.6+0.5B_C2C_longbench_e_50模型的结果，第三列是0.6+0.5B_C2C_longbench_e模型的结果。把f1的subject名称和f1值放在前几行，em的放在后几行，输出文件名为/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e_50/summary_comparison.csv
import json
import pandas as pd
json_file_1850 = '/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e_1850/Rosetta_longbench_generate_20260713_000700_summary.json'
json_file_300 = '/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e_300/Rosetta_longbench_generate_20260710_134433_summary.json'
output_csv = '/data/smy/lmy/C2C-compress-master/local/final_results/0.6+0.5B_C2C_longbench_e_1850/summary_comparison_1850.csv'
results_1850 = {}
results_300 = {}

with open(json_file_1850, 'r') as f:
    data_1850 = json.load(f)

    # 修正这里：定位到 JSON 的 'subjects' 字段
    for subject, metrics in data_1850['subjects'].items():
        results_1850[subject] = {}
        results_1850[subject]['score'] = metrics['score']
        results_1850[subject]['metric'] = metrics['metric']
  # Store the metric type (f1 or em) for each subject

with open(json_file_300, 'r') as f:
    data_300 = json.load(f)
    for subject, metrics in data_300['subjects'].items():
        results_300[subject] = {}
        results_300[subject]['score'] = metrics['score']
        results_300[subject]['metric'] = metrics['metric']  # Store the metric type (f1 or em) for each subject
# Create a DataFrame to hold the results
import pandas as pd

# 1. 用列表收集数据（比在循环中不断创建 DataFrame 高效数百倍）
rows = []
for subject in results_1850.keys():
    # 提取 metric 并做模糊兼容处理
    metric_val = results_1850.get(subject, {}).get('metric', '')
    
    rows.append({
        'subject': subject,
        '0.6+0.5B_C2C_longbench_e_1850': results_1850[subject]['score'],
        '0.6+0.5B_C2C_longbench_e_300': results_300.get(subject, {}).get('score', None),
        'metric': metric_val
    })

# 2. 一次性生成 DataFrame
df = pd.DataFrame(rows)

# 3. 按学科名称排序
df = df.sort_values(by='subject')

# 4. 分离不同的指标（修正 rouge 的匹配，兼容 "rouge_l"）
f1_subjects = df[df['metric'] == 'f1']
em_subjects = df[df['metric'] == 'em']
rouge_subjects = df[df['metric'].str.contains('rouge', na=False)] # 兼容 rouge_l 或 rouge

# 5. 合并并输出
final_df = pd.concat([f1_subjects, em_subjects, rouge_subjects])
final_df.to_csv(output_csv, index=False)
