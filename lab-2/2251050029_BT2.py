import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# BTH LT Pandas and Seaborn

# 1/ Pandas Series is essentially a one-dimensional array, equipped with an index which labels its entries.
# We can create a Series object, for example, by converting a list (called diameters) [4879,12104,12756,6792,142984,120536,51118,49528]
print("--- Câu 1 ---")
# Danh sách các đk
data = [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528]
diameters = pd.Series(data)
print(diameters)

# 2/ By default entries of a Series are indexed by consecutive integers, but we can specify a more meaningful index.
# The numbers in the above Series give diameters (in kilometers) of planets of the Solar System, so it is sensible to use names of the planet as index values:
# Index=[“Mercury”, “Venus”, “Earth”, “Mars”, “Jupyter”, “Saturn”, “Uranus”, “Neptune”]
print("\n--- Câu 2 ---")
planet_series = pd.Series(data, index=["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"])
print(planet_series)

# 3/ Find diameter of Earth?
print("\n--- Câu 3 ---")
print("Diameter of Earth:", planet_series["Earth"], "km")

# 4/ Find diameters from “Mercury” to “Mars” basing on data on 2/
# Tìm đường kính từ Mercury đến Mars
print("\n--- Câu 4 ---")
print("Diameters from Mercury to Mars:")
print(planet_series["Mercury":"Mars"])

# 5/  Find diameters of “Earth”, “Jupyter” and “Neptune” (with one command)?
print("\n--- Câu 5 ---")
print(planet_series[["Earth", "Jupyter", "Neptune"]])

# 6/ I want to modify the data in diameters. Specifically, I want to add the diameter of Pluto 2370.
# Saved the new data in the old name “diameters”.
print("\n--- Câu 6 ---")
diameters["Pluto"] = 2370
print(diameters)

# 7/ Pandas DataFrame is a two-dimensional array equipped with one index labeling its rows, and another labeling its columns.
# There are several ways of creating a DataFrame. One of them is to use a dictionary of lists. Each list gives values of a column of the DataFrame, and dictionary keys give column labels:
# “diameter”=[4879,12104,12756,6792,142984,120536,51118,49528,2370]
# “avg_temp”=[167,464,15,-65,-110, -140, -195, -200, -225]
# “gravity”=[3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]. Create a pandas DataFrame, called planets.
print("\n--- Câu 7 ---")
data = {
    "diameter": [4879, 12104, 12756, 6793, 142984, 120536, 51118, 49528, 2370],
    "avg_temp": [167, 464, 15, -65, -110, -140, -195, -200, -225],
    "gravity": [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
}
planets = pd.DataFrame(data,
                       index=["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
pd.DataFrame(data)
print(planets)

# 8/ Get the first 3 rows of “planets”.
print("\n--- Câu 8 ---")
print(planets.head(3))

# 9/ Get the last 2 rows of “planets”.
print("\n--- Câu 9 ---")
print(planets.tail(2))

# 10/ Find the name of columns of “planets”
print("\n--- Câu 10 ---")
print(planets.columns)

# 11/ Since we have not specified an index for rows, by default it consists of consecutive integers.
# We can change it by modifying the index by using the name of the corresponding planet. Check the index after modifying.
print("\n--- Câu 11 ---")
print(planets.index)

# 12/ How to get the gravity of all planets in “planets”?
print("\n--- Câu 12 ---")
print(planets["gravity"])

# 13/ How to get the gravity and diameter of all planets in “planets”?
print("\n--- Câu 13 ---")
print(planets[['gravity', 'diameter']])

# 14/ Find the gravity of Earth using loc?
print("\n--- Câu 14 ---")
print(planets.loc["Earth", "gravity"])

# 15/ Similarly, find the diameter and gravity of Earth?
print("\n--- Câu 15---")
print(planets.loc["Earth", ["diameter", "gravity"]])

# 16/ Find the gravity and diameter from Earth to Saturn?
print("\n--- Câu 16 ---")
print(planets.loc["Earth":"Saturn", ["gravity", "diameter"]])

# 17/ Check (using Boolean) all the planets in “planets” that have diameter >1000?
print("\n--- Câu 17 ---")
print(planets["diameter"] > 10000)

# 18/ Select all planets in “planets” that have diameter>100000?
print("\n--- Câu 18 ---")
print(planets[planets["diameter"] > 10000])

# 19/ Select all planets in “planets” that satisfying avg-temp>0 and gravity > 5
print("\n--- Câu 19 ---")
print(planets[(planets["avg_temp"] > 0) & (planets["gravity"] > 5)])

# 20/ Sort values of diameter in “diameters” in ascending order.
print("\n--- Câu 20 ---")
print(planets["diameter"].sort_values(ascending=True))

# 21/ Sort values of diameter in “diameters” in descending order.
print("\n--- Câu 21 ---")
print(planets["diameter"].sort_values(ascending=False))

# 22/ Sort using the “gravity” column in descending order in “planets”.
print("\n--- Câu 22 ---")
print(planets.sort_values(by="gravity", ascending=False))

# 23/ Sort values in the “Mercury” row.
print("\n--- Câu 23 ---")
print(planets.loc["Mercury"].sort_values())

# SEABORNS

# 1/ Seaborn is Python library for visualizing data. Seaborn uses matplotlib to create graphics,
# but it provides tools that make it much easier to create several types of plots.
# In particular, it is simple to use seaborn with pandas dataframes.
tips = sns.load_dataset("tips")

print("\n--- Câu 1 ---")
sns.set_style("whitegrid")  #Nền trắng, đường kẻ nhạt
g = sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2) #Tên trục x và y, rộng gấp 2 cao
g = (g.set_axis_labels("Tip", "Total bill(USD)").
     set(xlim=(0, 10), ylim=(0, 100)))
plt.title("title")
plt.show()

# 2/ Display name of datasets.
print("\n--- Câu 2 ---")
datasets_name = sns.get_dataset_names()
print(datasets_name)

# 3/ How can get a pandas dataframe with the data.
print("\n--- Câu 3 ---")
print(tips.head())

# 4/ How to produce a scatter plot showing the bill amount on the x axis and the tip amount on the y axis?
print("\n--- Câu 4 ---")
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()

# 5/ By default, seaborn uses the original matplotlib settings for fonts, colors etc. How to modify font=1.2 and color=darkgrid?
print("\n--- Câu 5 ---")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")

# 6/ We can use the values in the “day” column to assign marker colors. How?
print("\n--- Câu 6 ---")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()

# 7/ Next, we set different marker sizes based on values in the “size” column.
print("\n--- Câu 7 ---")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", size="size", sizes=(20, 200))
plt.show()

# 8/ We can also split the plot into subplots based on values of some column.
# Below we create two subplots, each displaying data for a different value of the “time” column
print("\n--- Câu 8 ---")
g = sns.FacetGrid(tips, col="time")  # Chia theo cột 'time'
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

# 9/ We can subdivide the plot even further using values of the “sex” column
print("\n--- Câu 9 ---")
g = sns.FacetGrid(tips, col="time", row="sex")  # Chia theo cả 'time' và 'sex'
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

# PANDAS (BT)
# Read file csv
gapminder = pd.read_csv("04_gap-merged.tsv", sep='\t')

# 1/ Show the first 5 lines of tsv file.
print("\n--- Câu 1 ---")
print(gapminder.head(5))

# 2/ Find the number of row and column of this file.
print("\n--- Câu 2 ---")
num_rows, num_columns = gapminder.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# 3/ Print the name of the columns.
print("\n--- Câu 3 ---")
print(gapminder.columns)

# 4/ What is the type of the column names?
print("\n--- Câu 4 ---")
print(type(gapminder.columns))

# 5/ Get the country column and save it to its own variable. Show the first 5 observations.
print("\n--- Câu 5 ---")
country_column = gapminder['country']
print(country_column.head(5))

# 6/ Show the last 5 observations of this column.
print("\n--- Câu 6 ---")
print(country_column.tail(5))

# 7/ Look at country, continent and year. Show the first 5 observations of these columns, and the last 5 observations.
print("\n--- Câu 7 ---")
country_continent_year = gapminder[['country', 'continent', 'year']]
print("The first 5 observations of these columns")
print(country_continent_year.head(5))
print("\nThe last 5 observations of these columns")
print(country_continent_year.tail(5))

# 8/ How to get the first row of tsv file? How to get the 100th row.
# Get the first row
print("--- Câu 8 ---")
# Cách 1
first_row_1 = gapminder.iloc[0]
# Cách 2
first_row_2 = gapminder.head(1)
print("The first row\n", first_row_1)
print("Cách 2\n", first_row_2)
# 9/ Try to get the first column by using a integer index. And get the first and last column by passing the integer index.
print("--- Câu 9 ---")
first_row = gapminder.iloc[:, 0]
print("Get the first column by using a integer index")
print(first_row)
first_and_last_column = gapminder.iloc[:, [0, -1]]
print("\nFirst and Last columns:")
print(first_and_last_column)

# 10/ How to get the last row with .loc? Try with index -1? Correct?
print("--- Câu 10 ---")
last_row = gapminder.loc[gapminder.index[-1]]
print(last_row)

# 11/ How to select the first, 100th, 1000th rows by two methods?
print("--- Câu 11 ---")
# Using .iloc
print("Using .iloc")
# Get the first row
first_row = gapminder.iloc[0]
# Get the 100th row
hunderedth_row = gapminder.iloc[99]
# Get the 1000th row
thounsandth_row = gapminder.iloc[999]
print("The first row\n", first_row)
print("\nThe 100th row\n", hunderedth_row)
print("\nThe 1000th row\n", thounsandth_row)

# Using .loc
# Get the first row
print("Using .loc")
first_row = gapminder.loc[gapminder.index[0]]
# Get the 100th row
hunderedth_row = gapminder.loc[gapminder.index[99]]
# Get the 1000th row
thounsandth_row = gapminder.loc[gapminder.index[999]]
print("The first row\n", first_row)
print("\nThe 100th row\n", hunderedth_row)
print("\nThe 1000th row\n", thounsandth_row)

# 12/ Get the 43rd country in our data using .loc, .iloc?
# Using .loc
print("Using .loc")
row_43rd = gapminder.loc[gapminder.index[42]]["country"]
print("The 43rd row")
print(row_43rd)
# Using .iloc
print("\nUsing .iloc")
row_43rd = gapminder.iloc[42]["country"]
print("The 43rd row")
print(row_43rd)

# 13/ How to get the first, 100th, 1000th rows from the first, 4th and 6th columns?
print("--- Câu 13 ---")
selected_data = gapminder.iloc[[0, 99, 999], [0, 3, 5]]
print(selected_data)

# 14/ Get first 10 rows of our data (tsv file)?
print("--- Câu 14 ---")
print("Get first 10 rows")
print(gapminder.head(10))

# 15/ For each year in our data, what was the average life expectation?
# Nhóm theo cột year, dùng mean để tính trung bình
print("--- Câu 15 ---")
print("For each year, the average life expectation")
print(gapminder.groupby('year')['lifeExp'].mean())

# 16/ Using subsetting method for the solution of 15/?
print("--- Câu 16 ---")
subset = gapminder[['year', 'lifeExp']]
print(subset.groupby('year').mean())

# 17/ Create a series with index 0 for ‘banana’ and index 1 for ’42’?
print("--- Câu 17 ---")
custom_series = pd.Series(["banana", 42], index=[0, 1])
print("Custom Series:")
print(custom_series)

# 18/ Similar to 17, but change index ‘Person’ for ‘Wes MCKinney’ and index ‘Who’ for ‘Creator of Pandas’?
print("--- Câu 18 ---")
custom_series = pd.Series(["Wes McKinney", "Creator of Pandas"], index=["Person", "Who"])
print("Custom Series:")
print(custom_series)

# 19/ Create a dictionary for pandas with the data as
#   ‘Occupation’: [’Chemist’, ’Statistician’],
#   ’Born’: [’1920-07-25’, ’1876-06-13’],
#   ’Died’: [’1958-04-16’, ’1937-10-16’],
#   ’Age’: [37, 61]
#   and the index is ‘Franklin’,’Gosset’ with four columns as indicated.
print("--- Câu 19 ---")
data = {
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-25', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'Age': [37, 61]
}
pandas_df = pd.DataFrame(data, index=['Franklin', 'Gosset'])
print(pandas_df)
