import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import altair as alt


#from streamlit_float import *
from streamlit_echarts import st_echarts,st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
#from streamlit_plotly_events import plotly_events


def scroll_to_section(section_id):
    st.markdown(
        f"""
        <script>
        document.getElementById('{section_id}').scrollIntoView({{
            behavior: 'smooth'
        }});
        </script>
        """,
        unsafe_allow_html=True
    )





st.set_page_config(layout="wide")
st.title("A mixed-method analysis to understand medicated weight loss")
tab1, tab2, tab3 = st.tabs(["Survey Analysis", "Patient Forum Analysis", "Integrated Analysis"])

#st.set_page_config(layout="wide")



with tab1:
    st.header("Survey Analysis")
    df_survey_pos = pd.read_excel("data/survey_positive_effect_open_responses.xlsx", sheet_name="Topic description")
    df_survey_neg = pd.read_excel("data/survey_negative_effect_open_responses.xlsx", sheet_name="Topic description")
    
    df_survey_pos = df_survey_pos[df_survey_pos["Captured on Reddit"].isin(["yes","no","somewhat"])].copy()
    df_survey_pos = df_survey_pos.sort_values(by=["Size"], ascending=True)
    df_survey_pos = df_survey_pos.dropna()

    df_survey_neg = df_survey_neg[df_survey_neg["Captured on Reddit"].isin(["yes","no","somewhat"])].copy()
    df_survey_neg = df_survey_neg.sort_values(by=["Size"], ascending=True)
    df_survey_neg = df_survey_neg.dropna()

    # Select sentiment
    sentiment_option = st.selectbox(
        "Which aspects of the patient experience do you want to study?",
        ("Positive","Negative"), key="test_sentiment_select")
    
    # Track sentiment toggle and reset selected topic if changed
    if 'last_sentiment_option' not in st.session_state:
        st.session_state.last_sentiment_option = sentiment_option
    elif sentiment_option != st.session_state.last_sentiment_option:
        st.session_state.selected_topic_plotly = None
        st.session_state.last_sentiment_option = sentiment_option

    if sentiment_option == "Positive":
        survey_df = df_survey_pos
        colour = "#00DB90"
    else:
        survey_df = df_survey_neg
        colour = "#FB5200"

    available_topics = set(survey_df["Description"])

    # Initialize session state
    if 'selected_topic_plotly' not in st.session_state:
        st.session_state.selected_topic_plotly = None

    # Create Plotly figure
    fig = go.Figure()
        
    fig.add_trace(go.Bar(
        x=survey_df["Size"],
        y=survey_df["Description"],
        orientation='h',
        marker_color = colour,
        text=survey_df["Size"],
        textposition='outside',
        texttemplate='%{text} counts',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br><extra></extra>'
    ))

    fig.update_layout(
        title=f"{sentiment_option} Topics mentioned in survey",
        xaxis_title="Count of respondents mentioning Topic",
        yaxis_title="Topics",
        height=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    selected_points = st.plotly_chart(
        fig, 
        use_container_width=True,
        selection_mode='points',
        on_select='rerun',
        key='plotly_chart_bar_survey'
    )

    if selected_points and selected_points['selection']['points']:
        point_idx = selected_points['selection']['points'][0].get('point_index')
        clicked_topic = survey_df.iloc[point_idx]["Description"]
        
        if clicked_topic != st.session_state.selected_topic_plotly:
            st.session_state.selected_topic_plotly = clicked_topic
            st.rerun()

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Reset Selection", key="plotly_reset"): # TO DO: check this works...
            st.session_state.selected_topic_plotly = None
            st.rerun()

    if st.session_state.selected_topic_plotly: # TO DO: check this works properly...
        if st.session_state.selected_topic_plotly in available_topics:
            st.markdown(f'<div id="survey_examples"></div>', unsafe_allow_html=True)
            scroll_to_section("survey_examples") # TO DO: check this works properly...
            st.markdown(f"### {st.session_state.selected_topic_plotly}")
            posts = survey_df[survey_df["Description"]==st.session_state.selected_topic_plotly].iloc[0]["Examples"]
            posts = posts.split("*")
            posts = [p.strip() for p in posts if len(p.strip())>0]
            
            st.markdown("**Example Responses**")
            for p in posts:
                st.markdown(f"""
                    <div style="
                        background-color: #CCCCCC;
                        border-left: 5px solid #636EFA;
                        padding: 20px;
                        margin: 15px 0;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    ">
                    <div style="
                        font-style: italic;
                        line-height: 1.6;
                        color: #495057;
                        font-size: 15px;
                    ">"{p}"</div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("Patient Forum Analysis")
    df_nlp = pd.read_excel("data/patient_forum_all_refined_topics.xlsx",sheet_name = "Exact_survey_topics")
    df_nlp = df_nlp[['Sentiment','New topic description','New_count', 'Paraphrased']]
    df_nlp['Log_count'] = np.log1p(df_nlp['New_count'])


    df_nlp_pos = df_nlp[df_nlp["Sentiment"]=="positive"].copy()
    df_nlp_pos = df_nlp_pos.sort_values(by = ["New_count"], ascending = True)
    df_nlp_pos = df_nlp_pos.dropna()
    # print(df_nlp_pos.head()['New_count'])
    # print(df_nlp_pos.size)

    df_nlp_neg = df_nlp[df_nlp["Sentiment"]=="negative"].copy()
    df_nlp_neg = df_nlp_neg.sort_values(by = ["New_count"], ascending = True)
    df_nlp_neg = df_nlp_neg.dropna()
    # print(df_nlp_neg.size)

    sentiment_option = st.selectbox(
        "Which aspects of the patient experience do you want to study?",
        ("Positive","Negative"), key="nlp_sentiment_select")

    if sentiment_option =="Positive":
        nlp_df = df_nlp_pos
        chart_title = "Positive Views about GLP-1 Usage"
        colour = "#00DB90"
    else:
        nlp_df = df_nlp_neg
        chart_title = "Negative Views about GLP-1 Usage"
        colour = colour = "#FB5200"

    available_topics_nlp = set(nlp_df["New topic description"])
    
    # Initialize session state
    if 'selected_topic_nlp' not in st.session_state:
        st.session_state.selected_topic_nlp = None
    
    fig_nlp = go.Figure()
    fig_nlp.add_trace(go.Bar(
        x=nlp_df["New_count"],
        y=nlp_df["New topic description"],
        orientation='h',
        marker_color=colour,
        text=nlp_df["New_count"],
        textposition='outside',
        texttemplate='%{text} counts',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br><extra></extra>'
    ))

    fig_nlp.update_layout(
        title=chart_title,
        xaxis_title="Count",
        yaxis_title="Topics",
        height=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    selected_points_nlp = st.plotly_chart(
        fig_nlp, 
        use_container_width=True,
        selection_mode='points',
        on_select='rerun',
        key='plotly_chart_nlp'
    )

    if selected_points_nlp and selected_points_nlp['selection']['points']:
        point_idx = selected_points_nlp['selection']['points'][0].get('point_index')
        clicked_topic_nlp = nlp_df.iloc[point_idx]["New topic description"]
        
        if clicked_topic_nlp != st.session_state.selected_topic_nlp:
            st.session_state.selected_topic_nlp = clicked_topic_nlp
            st.rerun()

    if st.session_state.selected_topic_nlp:
        if st.session_state.selected_topic_nlp in available_topics_nlp:
            point_idx = selected_points_nlp['selection']['points'][0].get('point_index')
            clicked_topic_nlp = nlp_df.iloc[point_idx]["New topic description"]
            st.markdown(f"### {clicked_topic_nlp}")
            
            posts = nlp_df[nlp_df["New topic description"]==st.session_state.selected_topic_nlp].iloc[0]["Paraphrased"]
            posts = posts.split("*")
            posts = [p.strip() for p in posts if len(p.strip())>0]
            
            st.markdown("**Example Responses**")
            for p in posts:
                st.markdown(f"""
                    <div style="
                        background-color: #CCCCCC;
                        border-left: 5px solid #636EFA;
                        padding: 20px;
                        margin: 15px 0;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    ">
                    <div style="
                        font-style: italic;
                        line-height: 1.6;
                        color: #495057;
                        font-size: 15px;
                    ">"{p}"</div>
                </div>
                """, unsafe_allow_html=True)
        

with tab3:
    st.header("Integrated Analysis")


    # Read survey dataset
    df_survey_pos = pd.read_excel("data/topic_ranking_differences.xlsx",sheet_name = "Positive") 
    df_survey_neg = pd.read_excel("data/topic_ranking_differences.xlsx",sheet_name = "Negative") 

    # Select sentiment
    sentiment_option = st.selectbox(
        "Which aspects of the patient experience do you want to study?",
        ("Positive","Negative"))

    if sentiment_option =="Positive":
        height = 800
        df = df_survey_pos
    else:
        df = df_survey_neg
        height = 1200

    data_items = df.to_dict("records")

    # Construct bump chart
    labels = list(df["Topic"])
    colours = [
            "#4B7CCC",  # Medium Blue
            "#F2668B",  # Pink
            "#03A688",  # Medium Teal
            "#FFAE3E",  # Amber
            "#B782B8",  # Purple
            "#A67F63",  # Chestnut
            "#0E8B92",  # Deep Teal
            "#D4AC2C",  # Bronze Yellow
            "#7E9F5C",  # Forest Green
            "#F7BCA3",  # Burnt Orange
            "#E63946",  # Red
            "#7DA9A7",  # Slate Blue
            "#457B9D",  # Steel Blue
            "#E094AC",  # Rose
            "#1D3557",  # Dark Blue
            "#2A9D8F",  # Teal Green
            "#B38A44",  # Antique Gold
            "#C68045",  # Copper
            "#264653",  # Charcoal Blue
        ]
    

    # colours =[
    #         "#003300",  # Very Dark Green
    #         "#013220",  # Dark Forest Green
    #         "#004225",  # British Racing Green
    #         "#014421",  # Deep Jungle Green
    #         "#06470C",  # Dark Leaf Green
    #         "#006400",  # DarkGreen (CSS)
    #         "#0B3D0B",  # Dark Fern Green
    #         "#1B4D3E",  # Myrtle Green
    #         "#204E27",  # Everglade
    #         "#254117",  # Dark Moss Green
    #         "#274E13",  # Verdun Green
    #         "#355E3B",  # Hunter Green
    #         "#3A5F0B",  # Sap Green
    #         "#3B5323",  # Olive Drab #7
    #         "#406C30",  # Rifle Green
    #         "#4A7023",  # Army Green
    #         "#556B2F",  # Dark Olive Green (CSS)
    #         "#384D48",  # Deep Eucalyptus
    #         "#223D26",  # Black Forest
    #         "#1C352D",  # Sacrificial Green
    # ]

    colours = colours[:len(labels)]

    mode_size = [15] * len(labels)
    line_size = [3] * len(labels)

    x_data = np.array([[1,2,3,4,5,6]]*len(labels))
    y_data = np.array([[data_items[i]["Overall Survey"],data_items[i]["Binge eating disorder Survey"],data_items[i]["Diabetes Survey"],data_items[i]["Obesity Survey"],data_items[i]["Weight loss more generally Survey"],data_items[i]["Patient Forum"]] for i in range(len(labels))])



    # Initialize session state
    if "selected_topic_plotly" not in st.session_state:
        st.session_state.selected_topic_plotly = None


    # Create Plotly figure
    fig_rank = go.Figure()



    # Add lines and points for each topic
    for i in range(len(labels)):
        # Add the line
        fig_rank.add_trace(go.Scatter(
            x=x_data[i], 
            y=y_data[i], 
            mode="lines+markers+text",
            name=labels[i],
            line=dict(color=colours[i], width=3),
            marker=dict(
                color=colours[i], 
                size=25,  # Larger markers to accommodate numbers
                line=dict(color="white", width=2)
            ),
            text=[f"{int(val)}" for val in y_data[i]],  # Show ranking numbers
            textposition="middle center",  # Center text on markers
            textfont=dict(
                size=15,
                color="white",
                family="Arial Black"
            ),
            hovertemplate="<b>%{fullData.name}</b><br>Ranking: %{y}<extra></extra>",
            showlegend=False
        ))

    # Update layout
    fig_rank.update_layout(
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            tickmode="array",
            tickvals=[1, 2,3,4,5,6],
            ticktext=["Survey", "Binge Eating Disorder","Diabetes","Obesity","General Weight Loss","Patient Forum"],
            tickfont=dict(
            family="Arial",
            size=16,
            color="rgb(82, 82, 82)",
            ),
            #range=[0.5, 2.5]  # Add padding on sides
        ),

        yaxis = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                autorange="reversed"  # Reverse so rank 1 is at top
        ),
        title = "Impact of mode of analysis on topic rankings",
        autosize=True,
        height=height,
        #width = 100,
        margin=dict(
            l=100,  # Left margin for topic labels
            r=50,  
            t=50,
            b=50
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Create annotations for topic labels
    annotations = []

    #Add topic labels on the left side only
    for i, (y_trace, label, color) in enumerate(zip(y_data, labels, colours)):
        # Left side label only (Survey ranking)
        annotations.append(dict(
            x=0.8, y=y_trace[0],
            xref="x", yref="y",
            text=f"<b>{label}</b>",
            font=dict(
                family="Arial",
                size=12,
                color=color
            ),
            showarrow=False,
            xanchor="right",
            yanchor="middle"
        ))


    fig_rank.update_layout(annotations=annotations)


    if "plotly_chart" in st.session_state:
        if st.session_state["plotly_chart"]["selection"]["points"]: #and len(st.session_state["plotly_chart"]["selection"]["points"])>0:
                    
            selected_index = int(st.session_state["plotly_chart"]["selection"]["points"][0].get("curve_number"))
            print(selected_index,st.session_state["plotly_chart"]["selection"])
            
            # Update the figure data
            for i in range(len(fig_rank.data)):
                if i == selected_index:
                    # Keep original styling for selected trace
                    fig_rank.data[i].line.color = colours[i]
                    fig_rank.data[i].marker.color = colours[i]
                    fig_rank.data[i].opacity = 1
                else:
                    # Grey out other traces
                    fig_rank.data[i].line.color ="#D3D3D3"
                    fig_rank.data[i].marker.color = "#D3D3D3"
                    fig_rank.data[i].opacity = 0.3

            # Update annotations for topic labels
            new_annotations = []

            #Add topic labels on the left side only
            for i, (y_trace, label) in enumerate(zip(y_data, labels)):
                # Left side label only (Survey ranking)
                if i!=selected_index:
                    new_annotations.append(dict(
                        x=0.8, y=y_trace[0],
                        xref="x", yref="y",
                        text=f"<b>{label}</b>",
                        font=dict(
                            family="Arial",
                            size=12,
                            color="#D3D3D3"
                        ),
                        showarrow=False,
                        xanchor="right",
                        yanchor="middle"
                    ))
                else:
                    new_annotations.append(dict(
                        x=0.8, y=y_trace[0],
                        xref="x", yref="y",
                        text=f"<b>{label}</b>",
                        font=dict(
                            family="Arial",
                            size=12,
                            color= colours[i]
                        ),
                        showarrow=False,
                        xanchor="right",
                        yanchor="middle"
                    ))

            fig_rank.update_layout(annotations=new_annotations)
        

        else:
            # Reset to original styling
            for i in range(len(fig_rank.data)):
                fig_rank.data[i].line.color = colours[i]
                fig_rank.data[i].marker.color = colours[i]
                fig_rank.data[i].opacity = 1

            # Reset annotation colours
            new_annotations = []
            for j, annotation in enumerate(fig_rank.layout.annotations):
                if annotation.x == 0.8:  # Left side topic labels
                    topic_index = j
                    if topic_index < len(labels):
                        annotation.font.color = colours[topic_index]
                new_annotations.append(annotation)

            fig_rank.update_layout(annotations=new_annotations)

    else:
        # Reset to original styling
        for i in range(len(fig_rank.data)):
            fig_rank.data[i].line.color = colours[i]
            fig_rank.data[i].marker.color = colours[i]
            fig_rank.data[i].opacity = 1

        # Reset annotation colours
        new_annotations = []
        for j, annotation in enumerate(fig_rank.layout.annotations):
            if annotation.x == 0.8:  # Left side topic labels
                topic_index = j
                if topic_index < len(labels):
                    annotation.font.color = colours[topic_index]
            new_annotations.append(annotation)

        fig_rank.update_layout(annotations=new_annotations)
        


    # Display the chart
    with st.container():
        selected_points = st.plotly_chart(fig_rank,
                use_container_width=True,
                selection_mode="points",
                on_select="rerun",
                key="plotly_chart",
                width = 100)

        print(selected_points)


