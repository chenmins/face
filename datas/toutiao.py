def process_line(line):
    fields = line.strip().split('_!_')
    # 提取 "Title", "Additional Info", 和 "Category label"
    title = fields[3]
    additional_info = fields[4]
    label = fields[2].replace('news_', '')
    # 合并这些字段，并将 "Category label" 添加到行的末尾
    new_line = f"{title}_!_{additional_info},{label}\n"
    return new_line

# File paths
input_file_path = 'toutiao_cat_data.txt'
output_file_path = 'toutiao_cat_data_mark.txt'

# Process each line in the file and save to a new file
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        outfile.write(process_line(line))
