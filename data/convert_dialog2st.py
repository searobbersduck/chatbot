# !/usr/bin/env python3
import os
import json

with open('dialog/dialog_txt.txt', 'w', encoding='utf8') as f:
    with open('dialog/dialog.txt', 'r', encoding='utf8') as f1:
        dialog_list = json.load(f1)
        for elem in dialog_list:
            f.write(elem['question'])
            f.write('\n')
            f.write(elem['answer'])
            f.write('\n')

