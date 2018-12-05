import csv
import sys

file = str(sys.argv[1])
count = int(sys.argv[2])

d = '_'
output = []
with open(file, 'r') as f:
    reader = csv.DictReader(f)
    for category in ['texture', 'object']:
        c = 0
        for line in reader:
            if line['category'] == category:
                output.append(str(int(line['unit'])-1) + d + line['category'] + d + line['label'] + d + line['score'])
                c += 1
            if c >= count:
                break
    c = 0
    for line in reversed(list(reader)):
        output.append(str(int(line['unit'])-1) + d + 'bad.' + line['category'] + d + line['label'] + d + line['score'])
        c += 1
        if c >= count:
            break
for line in output:
    assert(' ' not in line)
print(' '.join(output))