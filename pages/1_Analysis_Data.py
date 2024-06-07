import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd

def run():
    st.set_page_config(
        page_title="DataFrame Demo",
        page_icon="📊",
    )

    st.write("""This demo shows the features of COVID-19 as well as the visualization of each of the variables vs time in the span of 1 year.
    (Data courtesy of the [MoH COVID-19 Data](https://github.com/MoH-Malaysia/covid19-public.git).)""")

    #image
    st.image("covid_graph.jpg",width=500)

    #title
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        **👈 Select a demo from the sidebar** to see the graph demo for each models!
    """
    )


if __name__ == "__main__":
    run()

# Load the data with the correct date format
data = pd.read_csv('cases_malaysia.csv', parse_dates=['date'], dayfirst=True)
data['date'] = data['date'].dt.date

# Perform analysis
data = data.sort_values('date')
data['new_cases'] = data['confirmed'].diff()
data['new_S'] = data['susceptible'].diff()
data['new_I'] = data['infected'].diff()
data['new_R'] = data['recovered'].diff()
data['new_F'] = data['fatal'].diff()

# Title of the app
st.title('COVID-19 Data Analysis')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select Page', ['Overview', 'Detailed Analysis'])

# Overview Page
st.markdown("""
        - New cases and new S-I-R-F plots shows the daily changes in the dataset.
        - The **diff()** method in pandas is used to calculate the difference between consecutive rows in a DataFrame column.
        - By default, **diff()** computes the difference between the current row and the previous row.
        - This method helps us to analyze trends and patterns.
        """    
    )

if page == 'Overview':
    st.header('Confirmed Cases Over Time')
    fig = px.line(data, x='date', y='confirmed', title='Confirmed Cases Over Time')
    st.plotly_chart(fig)

    st.header('New Cases Over Time')
    fig = px.line(data, x='date', y='new_cases', title='New Cases Over Time')
    st.plotly_chart(fig)

# Detailed Analysis Page
elif page == 'Detailed Analysis':
    st.header('Data Table')
    st.subheader('1-year span data (1 Jan 2023 until 1 Jan 2024)')
    st.write(data)

    st.header('Interactive Plots')
    feature = st.selectbox('Select feature to plot', ['susceptible','infected', 'recovered', 'fatal','confirmed', 'new_cases', 'new_S', 'new_I', 'new_R', 'new_F'])

    # Debugging: Print the data to ensure it is available
    st.write(f"Selected feature: {feature}")
    st.write(data[['date', feature]])

    # Ensure the feature has no missing values
    data_to_plot = data[['date', feature]].dropna()

    # Plotting
    if not data_to_plot.empty:
        fig = px.line(data_to_plot, x='date', y=feature, title=f'{feature.capitalize()} Over Time')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected feature.")


