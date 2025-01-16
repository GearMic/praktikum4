from helpers import *

df = read_data_pd('lib/test.csv')
print(df)
df_formatted = format_df(df, ((1, 2),))
print(df_formatted)

print(find_last_zero_digit_index(0.001), find_last_zero_digit_index(0.1))