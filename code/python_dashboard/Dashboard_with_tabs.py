import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
dravet_data = pd.read_csv('NIH_DravetSyndrome_2014_2024.csv')

# Convert date columns to datetime format
date_columns = ['Award Notice Date', 'Project Start Date', 'Project End Date', 'Budget Start Date', 'Budget End Date']
for col in date_columns:
    dravet_data[col] = pd.to_datetime(dravet_data[col], errors='coerce')

# Convert numerical columns to appropriate data types
numerical_columns = ['Application ID', 'Fiscal Year', 'Total Cost', 'Total Cost IC']
for col in numerical_columns:
    dravet_data[col] = pd.to_numeric(dravet_data[col], errors='coerce')

# Filtering out entries where Fiscal Year is 0
filtered_data = dravet_data[dravet_data['Fiscal Year'] != 0]
filtered_data = filtered_data[filtered_data['Type'] != '139104']

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("NIH Funding Analysis for Dravet Syndrome and Pediatric Epilepsy Research", style={'textAlign': 'center'}),

    # Filters for selecting fiscal year and state
    html.Div([
        html.Div([
            html.Label("Select Fiscal Year(s):"),
            dcc.Dropdown(
                id='fiscal-year-dropdown',
                options=[{'label': year, 'value': year} for year in sorted(filtered_data['Fiscal Year'].unique())],
                value=sorted(filtered_data['Fiscal Year'].unique()),
                multi=True
            )
        ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '3%'}),

        html.Div([
            html.Label("Select State(s):"),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': state, 'value': state} for state in
                         sorted(filtered_data['Organization State'].unique())],
                value=sorted(filtered_data['Organization State'].unique()),
                multi=True
            )
        ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '1%'})
    ], style={'marginBottom': 50}),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploratory Analysis', value='tab-1'),
        dcc.Tab(label='Network Analysis', value='tab-2')
    ]),

    html.Div(id='tabs-content')
])


# Define the callbacks to update the figures based on the selected fiscal years and states
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('fiscal-year-dropdown', 'value'),
     Input('state-dropdown', 'value')]
)
def render_content(tab, selected_years, selected_states):
    # Filter data based on selections
    filtered_df = filtered_data[(filtered_data['Fiscal Year'].isin(selected_years)) &
                                (filtered_data['Organization State'].isin(selected_states))]

    # Funding Trends Over Time
    funding_trends = filtered_df.groupby('Fiscal Year')['Total Cost'].sum().reset_index()
    funding_trends['Total Cost'] = funding_trends['Total Cost'] / 1e6

    # Moving average for prediction
    model = ExponentialSmoothing(funding_trends['Total Cost'], trend='add', seasonal='add', seasonal_periods=4)
    fit = model.fit()
    forecast = fit.forecast(1)

    # Create a trace for the forecast
    forecast_trace = go.Scatter(
        x=[funding_trends['Fiscal Year'].max() + 1],
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(dash='dash', color='red')
    )

    # Line for moving average
    moving_avg = funding_trends['Total Cost'].rolling(window=3).mean()

    fig1 = go.Figure(data=[
        go.Bar(
            name='Total Funding',
            x=funding_trends['Fiscal Year'],
            y=funding_trends['Total Cost'],
            marker_color='skyblue',
            text=funding_trends['Total Cost'].round(2),
            hovertemplate='%{x}<br>Total Funding: $%{text}M'
        ),
        go.Scatter(
            name='Moving Average',
            x=funding_trends['Fiscal Year'],
            y=moving_avg,
            mode='lines',
            line=dict(color='orange')
        ),
        forecast_trace
    ])
    fig1.update_layout(
        title={'text': 'Total NIH Funding for Epilepsy Research by Fiscal Year', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Fiscal Year',
        yaxis_title='Total Funding ($ Millions)',
        height=400,
        template='plotly_white'
    )

    # Funding Distribution by Administering IC
    funding_by_ic = filtered_df.groupby('Administering IC')['Total Cost'].sum().sort_values(ascending=False).head(
        10).reset_index()
    funding_by_ic['Total Cost'] = funding_by_ic['Total Cost'] / 1e6
    fig2 = go.Figure(data=[
        go.Bar(
            name='Total Funding',
            x=funding_by_ic['Administering IC'],
            y=funding_by_ic['Total Cost'],
            marker_color='green',
            text=funding_by_ic['Total Cost'].round(2),
            hovertemplate='%{x}<br>Total Funding: $%{text}M'
        )
    ])
    fig2.update_layout(
        title={'text': 'Top 10 NIH Funding Sources for Dravet Syndrome Research by Administering IC', 'x': 0.5,
               'xanchor': 'center'},
        xaxis_title='Administering IC',
        yaxis_title='Total Funding ($ Millions)',
        height=400,
        template='plotly_white'
    )

    # Geographical Distribution
    funding_by_state = filtered_df.groupby('Organization State')['Total Cost'].sum().reset_index()
    funding_by_state['Total Cost'] = funding_by_state['Total Cost'] / 1e6
    fig3 = px.choropleth(
        funding_by_state,
        locations='Organization State',
        locationmode='USA-states',
        color='Total Cost',
        color_continuous_scale='Blues',
        scope='usa',
        labels={'Total Cost': 'Total Funds ($M)', 'Organization State': 'State'}
    )
    fig3.update_layout(
        title={'text': 'Total NIH Funding for Dravet Syndrome Research by State', 'x': 0.5, 'xanchor': 'center'}
    )

    # Funding Distribution by Activity
    activity_funding = filtered_df.groupby('Activity')['Total Cost'].sum().sort_values(ascending=False)
    top_5_activities = activity_funding.head(5)
    other_activities = activity_funding.iloc[5:].sum()
    top_5_activities['Other Types'] = other_activities
    top_5_activities = top_5_activities.reset_index()
    top_5_activities['Total Cost'] = top_5_activities['Total Cost'] / 1e6
    fig4 = px.pie(
        top_5_activities,
        names='Activity',
        values='Total Cost',
        title='Funding Distribution by Type of Activity',
        hole=0.3
    )
    fig4.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}: $%{value:,.2f}M'
    )
    fig4.update_layout(
        template='plotly_white',
        height=500,
        title={
            'text': 'Funding Distribution by Type of Activity',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    if tab == 'tab-1':
        return [
            html.Div(dcc.Graph(figure=fig1), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=fig2), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=fig3), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=fig4), style={'width': '50%', 'display': 'inline-block'})
        ]

    # Network Analysis
    elif tab == 'tab-2':
        # Top PIs by Funding and Projects
        pi_funding = filtered_df.groupby(['Contact PI Person ID', 'Contact PI / Project Leader', 'Organization State'])[
            'Total Cost'].agg(['sum', 'count']).reset_index()
        top_10_pis = pi_funding.sort_values(by='sum', ascending=False).head(10)
        top_10_pis.columns = ['PI Person ID', 'PI Name', 'State', 'Total Funding', 'Project Count']
        top_10_pis['Total Funding'] = top_10_pis['Total Funding'] / 1e6
        fig5 = go.Figure(data=[
            go.Table(
                header=dict(values=list(top_10_pis.columns),
                    fill_color='paleturquoise',
                    align='left'),
                cells=dict(values=[top_10_pis[col] for col in top_10_pis.columns],
                           fill_color='lavender',
                           align='left')
            )
        ])
        fig5.update_layout(
            title={'text': 'Top 10 PIs by Total Funding and Project Count', 'x': 0.5, 'xanchor': 'center'},
            height=500,
            template='plotly_white'
        )

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges
        for _, row in filtered_df.iterrows():
            main_pi = row['Contact PI Person ID']
            main_pi_name = row['Contact PI / Project Leader']
            other_pis = row['Other PI or Project Leader(s)'].split('; ')
            for other_pi in other_pis:
                if other_pi:  # Check if other_pi is not an empty string
                    G.add_edge(main_pi_name, other_pi)

        # Generate the network graph using Plotly
        pos = nx.spring_layout(G, seed=42)  # Layout for the network graph

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)
        )

        fig6 = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                             title='Network Graph of PIs and Collaborations',
                             titlefont_size=16,
                             showlegend=False,
                             hovermode='closest',
                             margin=dict(b=20, l=5, r=5, t=40),
                             annotations=[dict(
                                 text="Network Graph showing collaborations among PIs",
                                 showarrow=False,
                                 xref="paper", yref="paper",
                                 x=0.005, y=-0.002)],
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False))
                         )
        fig6.update_layout(
            title={'text': 'Network Graph of PIs and Collaborations', 'x': 0.5, 'xanchor': 'center'},
            height=500,
            template='plotly_white'
        )

        return [
            html.Div(dcc.Graph(figure=fig5), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(figure=fig6), style={'width': '50%', 'display': 'inline-block'})
        ]

    return html.Div()


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

