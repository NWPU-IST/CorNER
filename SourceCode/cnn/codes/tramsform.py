import csv
result_path = 'models/core'

csvFile = open(result_path + 'result.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(csvFile)
csvRow = []

f = open(result_path + 'result.txt', 'r', encoding='utf-8')

for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)
f.close()
csvFile.close()

'''
df = pd.read_csv(file)
    df.values
    data = df.as_matrix()
    data = list(map(list,zip(*data)))
    data = pd.DataFrame(data)
    data.to_csv('dataset/'+file,header=0,index=0)

'''
import pandas as pd

df = pd.read_csv(result_path + 'result.csv')
data = df.values # data是数组，直接从文件读出来的数据格式是数组
index1 = list(df.keys()) # 获取原有csv文件的标题，并形成列表
data = list(map(list, zip(*data))) # map()可以单独列出列表，将数组转换成列表
data = pd.DataFrame(data, index=index1) # 将data的行列转换
data.to_csv(result_path + 'result.csv', header=0)


