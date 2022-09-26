def load_data(filename, maxlen):
    """加载数据
    加载后每条数据格式：[text, (start, end, label), (start, end, label), ...]，意味着text[start:end + 1]是类型为label的实体。
    返回值：（数据，最大句子长度，实体类别，所有实体类别（包含B、I、E等位置信息））
    """
    D = []
    categories, categories_all = set(), set()
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                if not c:
                    continue
                char, flag = c.split('\t')
                d[0] += char
                if flag[0] in ['B', 'S']:
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                    categories_all.add(flag[:])
                elif flag[0] in ['I', 'E']:
                    d[-1][1] = i
                    categories_all.add(flag[:])
            if len(d[0]) <= maxlen:
                D.append(d)
    lenght_list = [len(line[0]) for line in D]
    length_sen = max(lenght_list)
    return D, length_sen, categories, categories_all


if __name__ == "__main__":
    D, length_sen, categories, categories_all = load_data("../dataset/train_data.txt")
    print("最大长度：%d，实体类别：%s，所有实体类别（包含B、I、E等位置信息）：%s"
          % (length_sen, str(categories), str(categories_all)))
