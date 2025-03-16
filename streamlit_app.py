import streamlit as st
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
from requests import Session
import os
import json
# import helpers
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import threading
import subprocess
from datetime import timedelta
import datetime

def run_streamlit_app(file_name):
    subprocess.run(['streamlit', 'run', file_name, '--server.port=8501'])

app = dash.Dash(__name__)

fig = go.Figure()

app.layout = [
    html.Div('This is the dash app.'),
    html.Iframe(
        src = 'http://localhost:8501',
        style = {
            'width': '100%',
            'height': '600px',
            'border': 'none'
        }
    ),
    dcc.Graph(
            id='scatter-plot',
            config={'displayModeBar': False},
            figure=fig
            )
    ]

lat_col = []
lon_col = []
df_filtered = dict({'Date': [],
                         'Game Id': [],
                          'Round Number': [],
                          'Country': [],
                          'Latitude': [],
                          'Longitude': [],
                          'Damage Multiplier': [],
                          'Opponent Id': [],
                          'Opponent Country': [],
                          'Your Latitude': [],
                          'Your Longitude': [],
                          'Opponent Latitude': [],
                          'Opponent Longitude': [],
                          'Your Distance': [],
                          'Opponent Distance': [],
                          'Your Score': [],
                          'Opponent Score': [],
                          'Map Name': [],
                          'Game Mode': [],
                          'Moving': [],
                          'Zooming': [],
                          'Rotating': [],
                          'Your Rating': [],
                          'Opponent Rating': [],
                          'Score Difference': [],
                          'Win Percentage': [],
                          '5k Border': [],
                          'Pano URL': []
                          })
metric_col = "Your Distance"

color_ = {"sequential": [
                    [0, 'rgb(191, 34, 34)'], 
                    [0.3, 'rgb(243, 10, 10)'],
                    [0.5, 'rgb(234, 174, 19)'],
                    [0.75, 'rgb(220, 231, 22)'],
                    [0.85, 'rgb(26, 227, 40)'],
                    [0.90, 'rgb(34, 187, 175)'],
                    [0.95, 'rgb(24, 111, 197)'],
                    [0.995, 'rgb(47, 47, 255)'],
                    [0.996, 'rgb(255, 255, 255)'],
                    [1, 'rgb(255, 255, 255)']
                    ]}

if metric_col == 'Distance':
    metric_col = 'Your Distance'
if metric_col == 'Your Score':
    color_ = {"sequential": [
    [0, 'rgb(191, 34, 34)'], 
    [0.3, 'rgb(243, 10, 10)'],
    [0.5, 'rgb(234, 174, 19)'],
    [0.75, 'rgb(220, 231, 22)'],
    [0.85, 'rgb(26, 227, 40)'],
    [0.90, 'rgb(34, 187, 175)'],
    [0.95, 'rgb(24, 111, 197)'],
    [0.995, 'rgb(47, 47, 255)'],
    [0.996, 'rgb(255, 255, 255)'],
    [1, 'rgb(255, 255, 255)']
    ]}
if metric_col == 'Your Distance':
    color_ = {"sequential": [
            [0, 'rgb(255, 255, 255)'], 
            [0.00025, 'rgb(255, 255, 255)'],
            [0.000255, 'rgb(85, 9, 213)'],
            [0.0025, 'rgb(35, 31, 119)'],
            [0.005, 'rgb(20, 102, 212)'],
            [0.025, 'rgb(17, 155, 166)'],
            [0.05, 'rgb(24, 111, 197)'],
            [0.1, 'rgb(47, 47, 255)'],
            [0.25, 'rgb(255, 255, 255)'],
            [1, 'rgb(255, 255, 255)']
            ]}
if metric_col == 'Score Difference':
    color_ = {"sequential": [
            [0, 'rgb(153, 26, 29)'], 
            [0.2, 'rgb(224, 69, 10)'],
            [0.4, 'rgb(222, 97, 65)'],
            [0.45, 'rgb(224, 183, 152)'],
            [0.475, 'rgb(211, 188, 165)'],
            [0.5, 'rgb(221, 221, 221)'],
            [0.525, 'rgb(193, 222, 192)'],
            [0.55, 'rgb(128, 213, 128)'],
            [0.6, 'rgb(68, 213, 86)'],
            [0.8, 'rgb(7, 156, 89)'],
            [1, 'rgb(7, 41, 156)']
            ]}

fig.add_trace(go.Scattermap(
        lat=lat_col,
        lon=lon_col,
        mode='markers',
        marker=go.scattermap.Marker(
            size=df_filtered["5k Border"],
            color="Black"
            ),
        text=df_filtered['Pano URL'],
        hoverinfo='text'
        ))

fig.add_trace(go.Scattermap(
    lat=lat_col,
    lon=lon_col,
    mode='markers',
    marker=go.scattermap.Marker(
        size=6,
        color=df_filtered[metric_col]
        ),
    text=df_filtered['Pano URL'],
    hoverinfo='text'
    ))

fig.update_layout(
    title=dict(text='Your guesses:'),
    autosize=True,
    hovermode='closest',
    showlegend=False,
    colorscale=color_,
    map=dict(
        bearing=0,
        pitch=0,
        zoom=0,
        style='light'
    ),
    )

if metric_col == 'Your Distance':
    fig.update_layout(coloraxis=dict(cmin=0, cmax=20000))
if metric_col == 'Score Difference':
    fig.update_layout(coloraxis=dict(
        cmin=-5000, cmax=5000))

fig.update_layout(map_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
st.plotly_chart(fig)

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'clickData')
)
def update_scatter_plot(clickData):
    fig.update_traces(hovertemplate='Click me!<extra></extra>')
# Add custom hover text for each point
    return fig

@app.callback(
    Output('scatter-plot', 'clickData'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData:
        point_index = clickData['points'][0]['pointIndex']
    url = df_filtered['Pano URL'][point_index]
    # Open the URL in a new tab
    import webbrowser
    webbrowser.open(url)

    return clickData

if __name__ == '__main__':
    threading.Thread(target = run_streamlit_app, daemon=True, kwargs = dict(file_name = 'streamlitembed.py')).start()
    app.run(debug = True)
