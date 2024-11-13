import matplotlib.pyplot as plt

# Example dictionary data
data = {
    'Segment A': 25,
    'Segment B': 15,
    'Segment C': 30,
    'Segment D': 20,
    'Segment E': 10
}

# Step 1: Extract labels and values from the dictionary
labels = list(data.keys())
values = list(data.values())

# Step 2: Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart from Dictionary Data')
plt.show()