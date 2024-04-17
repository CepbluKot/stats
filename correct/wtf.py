def divide_to_same_intervals(data: list, n_parts: int): 
    data = sorted(data)
    
    
    curr_interval = 1
    
    interval_is_ok = True
    
    prev_begin_id = 0
    while interval_is_ok:        
        curr_n_parts = 0
        begin_id = 0
        end_id = 1

        # all_intervals = []
        while end_id < len(data) and curr_n_parts <= n_parts:
            while end_id < len(data) and data[end_id] - data[begin_id] <= curr_interval:
                end_id += 1

            # print(curr_n_parts,data[begin_id-1], data[begin_id], '|',data[end_id-1],data[end_id], data[end_id+1])
            # all_intervals.append(data[begin_id:end_id])
            if end_id-1 < len(data) and data[end_id-1] - data[begin_id] != curr_interval:
                interval_is_ok = False
                break

            
            prev_begin_id = begin_id
            begin_id = end_id
            curr_n_parts += 1

        if not interval_is_ok:
            break

        elif end_id >= len(data) and curr_n_parts <= n_parts:
            if data[len(data) - 1] - data[prev_begin_id] == curr_interval:
                curr_interval += 1
        
            interval_is_ok = False
        

        else:
            curr_interval += 1


    final_interval_size = curr_interval - 1
    
    curr_n_parts = 0
    begin_id = 0
    end_id = 1
    all_intervals = []

    while end_id < len(data) and curr_n_parts < n_parts:
        while end_id < len(data) and data[end_id] - data[begin_id] <= final_interval_size:
            end_id += 1

        all_intervals.append(data[begin_id:end_id])
        begin_id = end_id 
        curr_n_parts += 1

    return all_intervals




import seaborn as sns
import numpy as np
import math


data = None
with open('/home/oleg/Documents/stats_lecs/correct/Москва_2021.txt') as file:
    data = [int(line.rstrip()) for line in file]


res = divide_to_same_intervals(data, 10)

for x in res:
    print (min(x)- max(x))
