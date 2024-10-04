from typing import Sequence, Mapping
import os

async def calculate_metrics(snippets: Sequence[Mapping[str, str]], query: str):
    a_f, b_f = set(), set()
    a = 0
    b = 0
    c = 0
    d = 0
    
    for snippet in snippets:
        if query in snippet['text'] and snippet['doc'] not in a_f:
            a += 1
            a_f.add(snippet['doc'])
        if query not in snippet['text'] and snippet['doc'] not in b_f:
            b += 1
            b_f.add(snippet['doc'])

    for file_name in os.listdir('test'):
        file_path = os.path.join('test', file_name)
        
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                content = file.read()
                if query in content and c < len(snippets):
                    c += 1
                if query not in content and d < len(snippets):
                    d += 1                
    c, d = c - a, d - b
    recall =  a / (a + c) if a + c != 0 else 0
    precision = a / (a + b) if a + b != 0 else 0
    accuracy = (a + d) / (a + b + c + d)
    error = (b + c) / (a + b + c + d)
    f_measure = 2 / (1/precision + 1/recall) if precision and recall else 0
    
    return recall, precision, accuracy, error, f_measure