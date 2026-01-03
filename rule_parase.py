
# python rule_parase.py
input_str = """
conj_32 :- not has_attr_1, has_attr_60, not has_attr_82.
label(3) :- not conj_32.
conj_85 :- has_attr_2, not has_attr_8, not has_attr_64.
label(2) :- conj_85.
conj_74 :- not has_attr_2, has_attr_93, not has_attr_99.
label(2) :- not conj_74.
conj_25 :- not has_attr_2, not has_attr_3.
label(1) :- conj_25.
"""


file_path = "/path/code/TELLER_e2e/rules.txt"

map_rule = {}
for i in range(1, 107):
    key = f"has_attr_{i}"
    value = f"P{i}"
    map_rule[key] = value

map_rule.update({
    " :- ": " = ",
    ".": "",
    ", ": " ∧ "
})

labels_map = {
    "确认偏差": 0,  # 67694
    "可用性启发式": 1,  # 41226
    "刻板印象": 2,  # 25891
    "光环效应": 3,  # 20129
    "权威偏见": 4,  # 19577
    "框架效应": 5,  # 15969
    "从众效应": 6,  # 13676
    "虚幻真相效应": 7,  # 12764
    "群体内偏爱": 8,  # 5685
    "单纯曝光效应": 9,  # 5477
    "对比效应": 10,  # 2879
    "过度自信效应": 11,  # 2651
    "损失厌恶": 12,  # 2299
    "结果偏差": 13,  # 1974
    "后见之明偏差": 14  # 1568
}


for label, index in labels_map.items():
    key = f"label({index})"
    value = f"L_{{{label}}}"
    map_rule[key] = value


for key, value in map_rule.items():
    input_str = input_str.replace(key, value)


# print(input_str)
# Split the input into clauses
clauses = input_str.split("\n")

labels = [
    "确认偏差", "可用性启发式", "刻板印象", "光环效应", "权威偏见",
    "框架效应", "从众效应", "虚幻真相效应", "群体内偏爱", "单纯曝光效应",
    "对比效应", "过度自信效应", "损失厌恶", "结果偏差", "后见之明偏差"
]

label_clauses = {label: [] for label in labels}

output_clause = []


for clause in clauses:
    matched = False
    for label in labels:
        if clause.startswith(f"L_{{{label}}}"):

            clause_content = clause.replace(f"L_{{{label}}} = ", "")
            label_clauses[label].append(clause_content)
            matched = True
            break
    if not matched:

        output_clause.append(clause)


for label, clauses_list in label_clauses.items():
    if clauses_list:
        concatenated_clause = f"L_{{{label}}} = " + " ∨ ".join(clauses_list)
        output_clause.append(concatenated_clause)

try:
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write("\n")
        for clause in output_clause:
            file.write(clause + '\n')
    print(f"内容已成功追加到文件：{file_path}")
except Exception as e:
    print(f"写入文件时出错：{e}")
