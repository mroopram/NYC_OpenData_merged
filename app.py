import pandas as pd
from dash import Dash, dcc, html, Input, Output, clientside_callback, callback, State, no_update
import plotly.express as px
import dash_bootstrap_components as dbc
import json
import warnings

warnings.filterwarnings(action="ignore")

# import local components
from components.navbar import navbar
from components.info_table import grid
from components.about import about_panel
from components.community_panel import community_panel
from components.city_view import city_panel
from components.community_panel_filter import community_panel_filter
from components.selBuilding_view import selBuilding_panel

#
# read data
# df = pd.read_csv("./data/NYC_housingOnly_v0.csv")
# df = df.astype({"ENERGY STAR Score": float})
# with open("./data/new-york-zip-codes-_1604.geojson") as f:
#     zip_geojson = json.load(f)
# df_cb = df[df.Borough == "QUEENS"].reset_index().drop(columns="index")
#

app = Dash(
    # The dbc CYBORG team is generalized as the main style theme for the application, seconded by the dark_theme.css within the assets folder
    external_stylesheets=[dbc.themes.CYBORG, "/assets/dark_theme.css"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [about_panel],
            title="Learn about ...",
            className="mt-1 ms-2 me-2 primary",
            style={
                "backgroundColor": "#060606",  # CYBORG-compatible dark background color for accordio item
                "color": "#white",  # CYBORG-compatible white text color for accordio item
            },
        ),
        dbc.AccordionItem(
            [city_panel],
            title="Your City ...",
            className="mt-1 ms-2 me-2 primary",
            style={
                "backgroundColor": "#060606",  # CYBORG-compatible dark background color for accordio item
                "color": "#white",  # CYBORG-compatible white text color for accordio item
            },
        ),
        dbc.AccordionItem(
            [community_panel],
            title="Your Community ...",
            className="mt-1 ms-2 me-2 primary",
            style={
                "backgroundColor": "#060606",  # CYBORG-compatible dark background color for accordio item
                "color": "#white",  # CYBORG-compatible white text color for accordio item
            },
        ),
        dbc.AccordionItem(
            [
                # html.P("This is the content of the second section"),
                # dbc.Button("Don't click me!", color="danger"),
                selBuilding_panel
            ],
            title="Your Building ...",
            className="mt-1 ms-2 me-2",
        ),
        #Chat Bot
        dbc.AccordionItem(
            [
        dbc.Row([
            dbc.Col([html.Div('Ask our LLM questions about the Energy Star Score in your neighborhood')], width=8),
        ]),
        dbc.Row([
            dbc.Col([dcc.Textarea(id='user-question', style={'width':400})], width=8),
            dbc.Col([html.Button('Submit', id='submit-btn')], width=8),
        ]),
        dbc.Row([
            dbc.Col([dcc.Markdown(id='response-div')], width=8),
        ]),     
            ],
            title="Q&A Bot ...",
            className="mt-1 ms-2 me-2",
        ),
    ], 
)
tab1_content = dbc.Container(
    [
        # dbc.Card(
        #     dbc.CardBody([filterPanel]),
        #     className="mt-3",
        # ),
        dbc.Card(
            dbc.CardBody([accordion]),
            className="mt-3 ",
        ),
    ],
    className=" mb-2",
    fluid=True,
)

tab2_content = dbc.Card(
    dbc.CardBody([grid]),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(
            tab1_content,
            label="Your Search",
            # style for the entire tab
            style={
                "backgroundColor": "#060606",  # CYBORG-compatible dark style for the entire tab background
                "color": "#white",
            },  # white text style for the entire tab
            # style for each tab
            tab_style={
                "backgroundColor": "#006BB6",  # NY blue color for tabs
                "color": "white",  # white text color for tabs
                "border": "2px solid #ffffff",  #
            },
            # style for the active tab,  which overwritten by CYBORG theme
            active_tab_style={
                "backgroundColor": "#d4e8f3",  # Baby blue for active tab
                "color": "black",  # black text color for active tab
                "border": "1.5px solid #ffffff",
            },  # white border for active tab
        ),
        dbc.Tab(
            tab2_content,
            label="NYC Info",
            # style for the entire tab
            style={
                "backgroundColor": "#060606",  # CYBORG-compatible dark style for the entire tab background
                "color": "#white",
            },  # white text style for the entire tab
            # style for each tab
            tab_style={
                "backgroundColor": "#006BB6",  # NY blue color for tabs
                "color": "white",  # white text color for tabs
                "border": "2px solid #ffffff",  #
            },
            # style for the active tab,  which overwritten by CYBORG theme
            active_tab_style={
                "backgroundColor": "#d4e8f3",  # Baby blue for active tab
                "color": "black",  # black text color for active tab
                "border": "1.5px solid #ffffff",
            },  # white border for active tab
        ),
        # dbc.Tab("This tab's content is never seen", label="Tab 3", disabled=True),
    ],
    className="m-3",
)

# color_mode_switch = html.Span(
#     [
#         dbc.Label(className="fa fa-moon", html_for="switch"),
#         dbc.Switch(
#             id="switch", value=True, className="d-inline-block ms-1", persistence=True
#         ),
#         dbc.Label(className="fa fa-sun", html_for="switch"),
#     ]
# )

app.layout = dbc.Container(
    [
        # dbc.Row([color_mode_switch]),
        dbc.Row([dbc.Col([navbar])]),
        # dbc.Row([dbc.Col([tabs])]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [tabs], className="mt-3 mb-3 d-flex justify-content-center"
                        ),
                    ],
                    className="m-3",
                    width=12,
                ),
                dbc.Col(
                    [],
                    # width=4,
                ),
            ],
        ),
    ],
    className="",
    # fluid=True,
)

# clientside_callback(
#     """
#     (switchOn) => {
#        document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
#        return window.dash_clientside.no_update
#     }
#     """,
#     Output("switch", "id"),
#     Input("switch", "value"),
# )

#connect to sql database
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

engine = create_engine('sqlite:///opendata_github_database.db', echo=False)
db = SQLDatabase(engine=engine)

#Create tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

TAVILY_API_KEY="your_key_here"
tavilySearchAPIWrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

tavily_tool = TavilySearchResults(api_wrapper=tavilySearchAPIWrapper, 
                                  description="Use for information about NYC Local Law 33/18 and denfitions related to Building Energy Scores in NYC.", 
                                  max_results=3)

tools = [tavily_tool]

#Create Agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

OPENAI_API_KEY="your_key_here"

llm = ChatOpenAI(model="o3-mini", api_key=OPENAI_API_KEY)

agent_executor = create_sql_agent(llm, db=db, agent_type="zero-shot-react-description", verbose=True, extra_tools=tools)


@callback(
    Output('response-div', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('user-question', 'value'),
    prevent_initial_call=True
)
def activate_llm(_, question):
    if len(question) > 0:
        response = agent_executor.invoke(question)
        return response['output']
    else:
        return no_update

if __name__ == "__main__":
    app.run_server(debug=True)
