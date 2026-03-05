import sys
with open('rf_out.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'MEJOR RESULTADO ENCONTRADO' in line:
            print(lines[i-1].strip())
            print(lines[i].strip())
            print(lines[i+1].strip())
            print(lines[i+2].strip())
            sys.exit(0)
