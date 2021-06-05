# -*- coding: utf-8 -*-
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["font.serif"] = ["SimHei"]


hotel_data = pd.read_csv("hotel_bookings.csv")
# print(hotel_data)
# print(hotel_data.info())
# print(hotel_data.isnull().sum()[hotel_data.isnull().sum()!=0])

#缺失值处理
hotel_data.drop("company", axis=1, inplace=True)
hotel_data["agent"].fillna(0, inplace=True)
hotel_data["children"].fillna(0.0, inplace=True)
hotel_data["country"].fillna("Unknown", inplace=True)

hotel_data["meal"].replace("Undefined", "SC", inplace=True)
# print(hotel_data["meal"].value_counts())

#异常值处理
zero_guests = list(hotel_data["adults"]
                   + hotel_data["children"]
                   + hotel_data["babies"]==0)

hotel_data.drop(hotel_data.index[zero_guests], inplace=True)
# print(hotel_data.shape)

data = hotel_data.copy()

# 整体入住情况分析
data["is_canceled"] = data["is_canceled"].astype(object)
data["is_canceled"].replace(0, "入住", inplace=True)
data["is_canceled"].replace(1, "取消", inplace=True)

total_booking = data["is_canceled"].value_counts()
fig = px.pie(total_booking,
             values=total_booking.values,
             names=total_booking.index,
             title="整体入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

#预订需求对比
booking = data["hotel"].value_counts()

fig = px.pie(booking,
             values=booking.values,
             names=booking.index,
             title="预订需求比较",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

#入住率对比
rh = data[data['hotel']=='Resort Hotel']
ch = data[data['hotel']=='City Hotel']

#fig = make_subplots(rows=1, cols=2)

rh_checkin = rh["is_canceled"].value_counts()
fig = px.pie(rh_checkin,
             values=rh_checkin.values,
             names=rh_checkin.index,
             title="度假酒店入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

ch_checkin = ch["is_canceled"].value_counts()
fig = px.pie(ch_checkin,
             values=ch_checkin.values,
             names=ch_checkin.index,
             title="城市酒店入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()


print(data["lead_time"].value_counts().head(10))
print(data['lead_time'].describe())


#只考虑入住数据
rh = data.loc[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
ch = data.loc[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]

# 提取出入住时长相关数据
rh["total_nights"] = rh["stays_in_weekend_nights"] + rh["stays_in_week_nights"]
ch["total_nights"] = ch["stays_in_weekend_nights"] + ch["stays_in_week_nights"]

num_nights_res = list(rh["total_nights"].value_counts().index)
num_bookings_res = list(rh["total_nights"].value_counts())
rel_bookings_res = rh["total_nights"].value_counts() / sum(num_bookings_res) * 100 # 转换为百分比

num_nights_cty = list(ch["total_nights"].value_counts().index)
num_bookings_cty = list(ch["total_nights"].value_counts())
rel_bookings_cty = ch["total_nights"].value_counts() / sum(num_bookings_cty) * 100 

res_nights = pd.DataFrame({"hotel": "Resort hotel",
                           "num_nights": num_nights_res,
                           "rel_num_bookings": rel_bookings_res})

cty_nights = pd.DataFrame({"hotel": "City hotel",
                           "num_nights": num_nights_cty,
                           "rel_num_bookings": rel_bookings_cty})

nights_data = pd.concat([res_nights, cty_nights], ignore_index=True)
plt.figure(figsize=(16, 8))
sns.barplot(x = "num_nights", y = "rel_num_bookings", hue="hotel", data=nights_data,
            hue_order = ["City hotel", "Resort hotel"])
plt.title("Length of stay", fontsize=16)
plt.xlabel("Number of nights", fontsize=16)
plt.ylabel("Guests [%]", fontsize=16)
plt.legend(loc="upper right")
plt.xlim(0,22)
plt.show()

print(data["days_in_waiting_list"].value_counts())

#考虑全部预订数据
rh = data.loc[(data["hotel"] == "Resort Hotel")]
ch = data.loc[(data["hotel"] == "City Hotel")]

# 提取出餐食预订相关数据
meal_res = list(rh["meal"].value_counts().index)
num_meal_res = list(rh["meal"].value_counts())
rel_meal_res = rh["meal"].value_counts() / sum(num_meal_res) * 100 # 转换为百分比

meal_cty = list(ch["meal"].value_counts().index)
num_meal_cty = list(ch["meal"].value_counts())
rel_meal_cty = ch["meal"].value_counts() / sum(num_meal_cty) * 100 

res_meals = pd.DataFrame({"hotel": "Resort hotel",
                           "meal_booking": meal_res,
                           "rel_num_bookings": rel_meal_res})

cty_meals = pd.DataFrame({"hotel": "City hotel",
                           "meal_booking": meal_cty,
                           "rel_num_bookings": rel_meal_cty})

meal_data = pd.concat([res_meals, cty_meals], ignore_index=True)
plt.figure(figsize=(10, 6))
sns.barplot(x = "meal_booking", y = "rel_num_bookings", hue="hotel", data=meal_data,
            hue_order = ["City hotel", "Resort hotel"])
plt.title("Meal booking options", fontsize=16)
plt.xlabel("Kind of meal", fontsize=16)
plt.ylabel("Guests [%]", fontsize=16)
plt.legend(loc="upper right")
plt.xlim(0,4)
plt.show()

#整体的月度人流量
plt.figure(figsize = (10,5))
data.groupby(['arrival_date_month'])['arrival_date_month'].count().plot.bar()


#考虑实际入住情况
#平均价格
data["adr_pp"] = data["adr"] / (data["adults"] + data["children"])
full_data_guests = data.loc[data["is_canceled"] == 0]
rh = data.loc[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
ch = data.loc[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]

# grab data:
room_prices_mothly = full_data_guests[["hotel", "arrival_date_month", "adr_pp"]].sort_values("arrival_date_month")

# order by month:
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
room_prices_mothly["arrival_date_month"] = pd.Categorical(room_prices_mothly["arrival_date_month"], categories=ordered_months, ordered=True)

# barplot with standard deviation:
plt.figure(figsize=(12, 8))
sns.lineplot(x = "arrival_date_month", y="adr_pp", hue="hotel", data=room_prices_mothly, 
            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))
plt.title("Room price per night and person over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Price [EUR]", fontsize=16)
plt.show()

#月度人流量
resort_guests_monthly = rh.groupby("arrival_date_month")["hotel"].count()
city_guests_monthly = ch.groupby("arrival_date_month")["hotel"].count()

resort_guest_data = pd.DataFrame({"month": list(resort_guests_monthly.index),
                    "hotel": "Resort hotel", 
                    "guests": list(resort_guests_monthly.values)})

city_guest_data = pd.DataFrame({"month": list(city_guests_monthly.index),
                    "hotel": "City hotel", 
                    "guests": list(city_guests_monthly.values)})
full_guest_data = pd.concat([resort_guest_data,city_guest_data], ignore_index=True)
# order by month:
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
full_guest_data["month"] = pd.Categorical(full_guest_data["month"], categories=ordered_months, ordered=True)
# Dataset contains July and August date from 3 years, the other month from 2 years. Normalize data:
full_guest_data.loc[(full_guest_data["month"] == "July") | (full_guest_data["month"] == "August"),
                    "guests"] /= 3
full_guest_data.loc[~((full_guest_data["month"] == "July") | (full_guest_data["month"] == "August")),
                    "guests"] /= 2
plt.figure(figsize=(12, 8))
sns.lineplot(x = "month", y="guests", hue="hotel", data=full_guest_data, 
             hue_order = ["City hotel", "Resort hotel"], size="hotel", sizes=(2.5, 2.5))
plt.title("Average number of hotel guests per month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of guests", fontsize=16)
plt.show()