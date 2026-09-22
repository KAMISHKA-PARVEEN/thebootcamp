import numpy as np
arr = np.arange(1, 101)
print(arr)
print(arr**2)

#every time same random numbers

#step 01
players = np.array(["Virat", "Rohit", "Shami", "Hardik"])
matches = np.array(["IPL2021", "IPL2022", "IPL2023", "IPL2024"])

np.random.seed(42)

scores = np.random.randint(20, 101, size=(4, 4))

print("Players:", players)
print("Matches:", matches)
print("Scores:\n", scores)

row_sum = scores.sum(axis=0)
col_sum = scores.sum(axis=1)

#highest score for each player
max_idx = np.argmax(scores, axis=1)
print(max_idx)
print("Highes score of each player", scores[np.arange(len(players)),max_idx])
print("Match:", matches[max_idx])

#highest overall
max_overall = np.argmax(row_sum)
print(row_sum)
print(row_sum[max_overall])

#mean
print(np.mean(scores))
print(np.mean(scores, axis=1))
print(np.mean(scores, axis=0))

#median
print(np.median(scores))
print(np.median(scores, axis=1))
print(np.median(scores, axis=0))



#std deviation
print("Standard deviation:", np.std(scores))
print("Standard deviation:", np.std(scores, axis = 1))
print("Standard deviation:", np.std(scores, axis = 0))


import matplotlib.pyplot as plt
time = np.linspace(0, 10, 500)#500 equidistant
amplitude = np.exp(0.5*time)*np.sin(2*np.pi*time)

#create the canvas (figure)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi = 100)

#plot the daatt along axis
ax.plot(time, amplitude, color="crimson", linewidth=2, label='Sensor A Displacement')
ax.set_title('Damped Harmonic Response of Structure', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Displacement (mm)', fontsize=12)
# Enable engineering gridlines
ax.grid(True, linestyle='--', alpha=0.6)
# Configure the Legend
ax.legend(loc='upper right', frameon=True, shadow=True)

#plt.show()

from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
x = np.linspace(0, 4*np.pi, 200)
line, = ax.plot(x, np.sin(x), color='crimson', lw=2)

line.set_ydata(np.cos(x))
plt.show()

import pandas as pd
import numpy as np

mydataset = {
    'cars' : ['BMW', 'Volvo', "Fortuner", "Volvo"],
    "passings": [3, 7, 5, 8]
}

myvar = pd.DataFrame(mydataset)

print(myvar)

data = {
    'Visitor_ID': [f'V{1000 + i}' for i in range(20)],
    'Name': ['ABHISHEK SANDEEP      ZADE     ', 'ARNAV AJAY DESHPANDE.. ', '.. .. ASHWINI LALCHAND MUNDAWARE/ ', 'GAYATRI SURESH GAIKWAD', 'HARSHADA GANESH CHAUDHARI',
             'VAIBHAVI HARISHWAR PATIL', np.nan, 'VISHAKHA PUNDLIK JADHAV', 'YASH BHARAT SOLUNKE/', 'VIVEK SANTOSH KHANDWE',
             'VISHAKHA PUNDLIK JADHAV', 'Tanushree chhanwal', 'Shruti jaiswal', 'Shriyash Sulakhe', 'YASH BHARAT SOLUNKE',
             np.nan, 'ARNAV AJAY DESHPANDE', 'RUTUJA SANTOSH THOTE', 'ROHIT DILIP BILWAL', 'RITESH SHIVAJI BAIRAGI'],
    'Age': [25, 23, 22, np.nan, 21, 25, 24, 24, 28, np.nan,
            22, 23, 25, 27, 25, 30, 31, 26, 19, 19],
    'Ticket_Price': [500, 750, 500, 1000, np.nan, 500, 700, 650, 750, 1000,
                      500, 800, np.nan, 750, 500, 700, 900, 850, 750, np.nan],
    'Check_In_Time': ['10:00 AM', '10:30 AM', '11:00 AM', '11:30 AM', np.nan,
                       '10:00 AM', '12:00 PM', '12:30 PM', '01:00 PM', '01:30 PM',
                       '11:00 AM', '02:00 PM', np.nan, '02:30 PM', '10:00 AM',
                       '12:00 PM', '03:00 PM', '03:30 PM', '04:00 PM', np.nan],
    'City':['Delhi','Aurangabad', 'Mumbai','Bombay','New Delhi','NDL','Chennai','Chenai','Chennaai','Bangalore',
            'Delhi','Pune', 'New Delhi','Bombay','New Delhi','NDL','Indore','Bangalore','Ujjain','Bangalore'],
    'State': ['Delhi','Maharastra', 'Maharastra','Maharastra','Delhi','Delhi','Tamilnadu','Tamilnadu','Tamilnadu','Karnataka',
            'Delhi','Maharastra', 'Delhi','Maharastra','Delhi','Delhi','Madhya Pradesh','Karnataka','Madhya Pradesh','Karnataka']}

mydata = pd.DataFrame(data)
#print(mydata)

#print(mydata.head())
#print(mydata.tail())

mydata.to_csv("kumbh_mela_data.csv", index=False)

df = pd.read_csv("kumbh_mela_data.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())