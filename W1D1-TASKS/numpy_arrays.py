import numpy as np

marks = np.array([
    [70, 80, 90],
    [60, 75, 85],
    [88, 92, 79],
    [55, 65, 70]
])

print("Shape:", marks.shape)
print("Data type:", marks.dtype)
print("Size:", marks.size)

# Slicing
print("Slicing:", marks[:2, -1])

# Student averages
student_avg = marks.mean(axis=1)
print("Student averages:", student_avg)

# Subject averages
subject_avg = marks.mean(axis=0)
print("Subject averages:", subject_avg)

# Broadcasting
grace_marks = marks + 5
print("Grace marks:")
print(grace_marks)

# Weighted scores
weights = np.array([0.3, 0.3, 0.4])
weighted_scores = np.dot(marks, weights)
print("Weighted scores:", weighted_scores)

# Reshape
numbers = np.arange(12)
reshaped = numbers.reshape(3, 4)
print("Reshaped array:")
print(reshaped)

print("Slicing:", marks[:2,-1])

student_avg = marks.mean(axis=1)

print("Student averages:", student_avg)

subject_avg = marks.mean(axis=0)

print("Subject averages:", subject_avg)

grace_marks = marks + 5

print("Grace marks:")
print(grace_marks)

weights = np.array([0.3, 0.3, 0.4])

weighted_scores = np.dot(marks, weights)

print("Weighted scores:", weighted_scores)

numbers = np.arange(12)

reshaped = numbers.reshape(3, 4)

print("Reshaped array:")
print(reshaped)

