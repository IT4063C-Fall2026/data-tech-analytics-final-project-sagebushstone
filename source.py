#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# The purchasing of a home is a complicated process. With this project, I want to make the process simpler—or at least enumerate all one's options better so that an informed decision can be made by the buyer. The Cincinnati housing market is complex and this project aims to summarize the market and infer home pricing validity.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# The broad question this project aims to answer is: what does the Cincinnati housing market look like at present?
# 
# I plan to break this down into the following questions:
# What is the average home price in Cincinnati? The median?
# What is the distribution of home prices? Distribution of square footage?
# How do home prices and square footage correlate, if at all?
# Given a house's square footage and other factors, can I predict what a fair home price would be?
# Which areas in Cincinnati are safest for buying homes?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# I want to use box plots to demonstrate the distribution of housing prices and size of house. Bar graphs for price by neighborhood. Scatter plot for home price and square footage. I want to use machine learning to define an appropriate house price based on parameters; I think classification would work if I assigned a range of prices, or maybe regression for more accuracy. A map displaying crime reports or a bar chart with crimes by neighborhood would also be interesting. With these visualizations, I should be able to draw some useful information from them.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# I plan to use primarily a dataset from the Rentcast API (/listings/sale endpoint), which gives a lot of the same information as MLS systems. I will limit the API call to just Cincinnati and loop through to get all the offerings in the area.
# I also plan to use the Cincinnati CPD Crime Reports dataset, which I will download directly from the website and put in my assets folder.
# To be able to compare these, I want to utilize neighborhood as an indicator for crime level against housing location. The Rentcast API doesn't have the neighborhood (i.e. Avondale, Over the Rhine, Madisonville, etc.) so I will need to bring in an additional API for that. I will run each Rentcast listing address against the Cincinnati provided ArcGIS API service and plot each lat/long for the house against the geojson provided by that API. This will give me neighborhoods within the city limits, so Fairfax for instance would not be provided. However, I should still get a good number of usable results, and the crime dataset also has the same limitations about location.
# 
# Please also note that the Rentcast API is a paid API, so I need to be careful about how much I use it. I will include my request code in this document but will save the data to a CSV so that I only have to call the API one time. This will also make it easier for me to run each address against the ArcGIS API.
# 
# Overall, the model would essentially be as such: [Rentcast listings --M:1-- Geocoding neighborhood --1:M-- Crime reports] so the ArcGIS API acts as a cross-reference table between the listings and the crimes.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# I plan to use a variety of analysis techniques in the development of this project. I will primarily use quantitative techniques to summarize and explain the current Cincinnati housing market. I will also develop predictive models for identifying future trends. The Rentcast API, Cincinnati crime reports dataset, and Cincinnati ArcGIS API mesh well together to be able to create a descriptive picture of the housing market. There is good quantitative data already stored in the datasets (like price or square footage) that can be analyzed further.

# In[5]:


# importing my libraries
import pandas as pd
import requests
import os
from dotenv import load_dotenv, find_dotenv
import geopandas as gpd


# In[38]:


# API url for rentcast
rentcastURL = "https://api.rentcast.io/v1/listings/sale?city=Cincinnati&state=OH&limit=500&includeTotalCount=true"
load_dotenv(find_dotenv())
headers = {
    "X-Api-Key": os.getenv("rentcast-api")
}
offset = 0
null = None
lastPage = False

# looping through each page of the API and appending it to a CSV
while lastPage == False:
    # query the api
    response = requests.get(rentcastURL + f"&offset={offset}", headers=headers)
    df = pd.json_normalize(response.json())

    # include headers for the first response and not for the subsequent responses
    if offset != 0:
        df.to_csv("assets/RentCastOutput.csv", mode='a', header=False, index=False, sep=",")
    else:
        df.to_csv("assets/RentCastOutput.csv", header=True, index=False, sep=",")

    # increase the offset by the limit (500) to go to the next page
    offset += 500

    # if offset larger than the total count of listings, end the loop
    if offset >= int(response.headers['x-total-count']):
        lastPage = True



# In[43]:


df = pd.read_csv("assets/RentCastOutput.csv")

df = df.loc[:, ~df.columns.str.startswith("history.")]

df.to_csv(
    "assets/RentCastOutputCleaned.csv",
    index=False
)


# In[ ]:


# getting the geojson for each neighborhood in Cincinnati
# this is my alternative to the Geocoding API I was originally going to use because that would take way too long to run to be feasible
# using the Cincinnati ArcGIS service
url = (
    "https://services.arcgis.com/JyZag7oO4NteHGiq/"
    "ArcGIS/rest/services/Open_Data_Feature_Collection/"
    "FeatureServer/12/query"
)

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "f": "geojson"
}

response = requests.get(url, params=params)
response.raise_for_status()

# write the geojson to a file
with open("assets/cincinnatiNeighborhoods.geojson", "wb") as f:
    f.write(response.content)


# In[ ]:


# load neighborhoods and listings
neighborhoods = gpd.read_file("assets/cincinnatiNeighborhoods.geojson")
listings = pd.read_csv("assets/RentCastOutputCleaned.csv")

# setting the lat/long for each listing in the appropriate coordinate system
geoListings = gpd.GeoDataFrame(
    listings,
    geometry=gpd.points_from_xy(
        listings["longitude"],
        listings["latitude"]
    ),
    crs="EPSG:4326"
)

neighborhoods = neighborhoods.to_crs(geoListings.crs)

# join together the neighborhoods and the listings
matched = gpd.sjoin(
    geoListings,
    neighborhoods[["SNA_NAME", "geometry"]],
    how="left",
    predicate="within"
)
matched = matched.rename(columns={'SNA_NAME': 'neighborhood'})

# output the listings with neighborhoods to a csv
matched.to_csv("assets/AllListings.csv")


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# https://services.arcgis.com/JyZag7oO4NteHGiq/ArcGIS/rest/services/Open_Data_Feature_Collection/FeatureServer/12
# 
# https://developers.rentcast.io/reference/introduction

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

