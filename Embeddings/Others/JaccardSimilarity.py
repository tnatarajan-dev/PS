

def jaccard_similarity(x,y):
    intersection = len(set.intersection(*[set(x), set(y)]))
    union = len(set.union(*[set(x), set(y)]))
    return intersection/float(union)


#customers = ["The bottle is empty", "There is nothing in the bottle"]
customers = [
    "Customer A is in high risk category",
    "Customer B though in high risk category will not delinquent in the next 6 months",
    "Customer C is not in high risk category but will delinguent in the next 3 months",
    "Customer D is very risk category"
]
customers = [sent.lower().split(" ") for sent in customers]
print("First and Second")
print(jaccard_similarity(customers[0], customers[1]))
print("Second and First")
print(jaccard_similarity(customers[1], customers[0]))
print("First and Third")
print(jaccard_similarity(customers[0], customers[2]))
print("Third and First")
print(jaccard_similarity(customers[2], customers[0]))
print("Second and Third")
print(jaccard_similarity(customers[1], customers[2]))
print("Third and Second")
print(jaccard_similarity(customers[2], customers[1]))
print("First and Fourth")
print(jaccard_similarity(customers[0], customers[3]))
print("Fourth and First")
print(jaccard_similarity(customers[3], customers[0]))



