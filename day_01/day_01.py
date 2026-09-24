print("Welcome SRM students")

x = 5
y = 3.14
name = "Kamishka Parveen"
is_learning = True

print("Innteger example", x)
print("Float example", y)
print("String example", name)
print("Boolean example", is_learning)

names = ["kamishka", "ayushi", "rahsi"]

str = "Bootcamp on AI"
lst = str.split()
print(lst)
#reverse order
for i in range(len(lst)-1, -1, -1):
    print(lst[i])

#original order
for i in range(len(lst)):
    print(lst[i])


#Assignment 01
numbers = [1, 9, 8, 3, 4, 6, 5, 2, 0]
print("Even numbers")
for x in range(len(numbers)):
    if numbers[x] % 2 == 0:
        print(numbers[x])


sentence = "This is the first day of Bootcamp"
reversed_sen = sentence[::-1]
print(reversed_sen)

print(" ".join(sentence.split()[::-1]))