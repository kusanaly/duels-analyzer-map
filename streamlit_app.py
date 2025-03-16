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

fig = go.Figure()

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
    url = df['Pano URL'][point_index]
    # Open the URL in a new tab
    import webbrowser
    webbrowser.open(url)

    return clickData

if __name__ == '__main__':
    threading.Thread(target = run_streamlit_app, daemon=True, kwargs = dict(file_name = 'streamlitembed.py')).start()
    app.run(debug = True)
