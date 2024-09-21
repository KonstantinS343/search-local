import threading
import inotify.adapters
import requests
from pathlib import Path
import os
from collections.abc import Sequence


def watch_files(file: str, address: str):
    i = inotify.adapters.Inotify()
    i.add_watch(file)  

    print(f"Started watching: {file}")

    for event in i.event_gen(yield_nones=False): 
        (_, type_names, path, filename) = event
        
        if 'IN_MOVED_FROM' in type_names:
            requests.delete('http://' + address + '/sync/delete/', params={'file': os.path.join(path, filename)})
                
        
        if 'IN_CREATE' in type_names:
            if Path(os.path.join(path, filename)).is_file():
                print(f"New file detected: {os.path.join(path, filename)}")
                t = threading.Thread(target=watch_files, args=(os.path.join(path, filename), address))
                t.start()
            elif Path(os.path.join(path, filename)).is_dir():
                print(f"New directory detected: {os.path.join(path, filename)}, adding to watch.")
                t = threading.Thread(target=watch_files, args=(os.path.join(path, filename), address))
                t.start()
        
        if 'IN_CLOSE_WRITE' in type_names and not Path(path).is_dir():
            try:
                with open(path, 'r') as f:
                    file_content = f.read()
            except FileNotFoundError:
                continue
            requests.post('http://' + address + '/sync/', json={'filename': path, 'content': file_content})


def validate_source(folders: Sequence[str], address: str):
    valid_folders = set()
    exists_folders = []
    for item in folders: 
        if Path(item.strip()).exists():
            exists_folders.append(item.strip()) 
            valid_folders.add(item.strip())
    
    for item in exists_folders:
        if Path(item.strip()).is_dir(): 
            for root, dirs, files in os.walk(item.strip()):
                for file in files:
                    with open(os.path.join(root, file), 'r') as f:
                        file_content = f.read()
                    requests.post('http://' + address + '/sync/', json={'filename': os.path.join(root, file), 'content': file_content})
                    valid_folders.add(os.path.join(root, file))
                for dir in dirs:
                    valid_folders.add(os.path.join(root, dir))
                    
    return list(valid_folders)
                         

def _main():
    address = None
    with open('client/server', 'r') as file:
        address = file.read()
    
    config_files = requests.get('http://' + address + f':2000/config/')
    
    if config_files.status_code != 200:
        print(config_files.content)
        exit()

    files = validate_source(config_files.json().get('paths'), address)

    threads = []
    for file in files:
        t = threading.Thread(target=watch_files, args=(file, address))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == '__main__':
    _main()