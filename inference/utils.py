import requests
import tqdm 

def part_number_gen(num_parts):
    """
    Quick helper funtion to iterate link indexes in Coraal dataset
    """
    parts_idx = list(range(1, num_parts+1))
    parts = []
    for i in parts_idx:
        i = str(i)
        if len(i) == 1:
            parts.append(f"0{i}")
        else:
            parts.append(i)
    return parts

def download(link, path_to_store, progress_bar=False):
    response = requests.get(link, stream=True)
    if response.status_code == 200:
        total = int(response.headers.get("content-length", 0))

        if progress_bar:
            bar = tqdm(total=total, 
                    unit="iB", 
                    unit_scale=True, 
                    unit_divisor=1024)
            
        with open(path_to_store, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)

                if progress_bar:
                    bar.update(size)

    else:
        print(f"Failed to download {path_to_store}")

