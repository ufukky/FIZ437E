from numpy import longfloat


prt = ''

for _e in range(1,51):
    epochLimit = _e*10
    prt = prt + f"{epochLimit},"
#print(prt)

file1 = open('epoch10n.txt', 'r')
Lines = file1.readlines()

kacc = ''
k = 9
for line in Lines:
    if f"k:{k}  acc:" in line:
        acc = float(line.replace(f'k:{k}  acc:',''))
        kacc = kacc + str(acc) + ','
print(kacc)