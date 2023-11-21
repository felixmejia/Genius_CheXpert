
A={0: 0.5291077494621277,
 1: 0.5577911734580994,
 2: 0.33677369356155396,
 3: 0.3848506808280945,
 4: 0.3956454396247864}

res = 0
for val in A.values(): 
    res += val 
    print(val)
  
# using len() to get total keys for mean computation 
res = res / len(A) 
print("AVG={}", res)
