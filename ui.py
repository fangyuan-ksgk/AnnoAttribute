# Develop UI for annotation & preference collections
# Eventually, we will have a pool of Tips.
# -- Some of the Tips are Objective  Tips
# -- Some of the Tips are Subjective Tips
# -- We will fit a preference matrix to each customer

# Currently, first step is to develop a UI for the annotation of the Tips (in future, this will be either for subjective preference, and objective annotation)
# -- decomposition will matter || Collaborate with Rafheal, and the phycologist!
import streamlit as st
import pandas as pd 
import numpy as np 
import glob, json, os
from typing import Tuple, List, Dict

# Dataframe is simply a better option
# for each convseration pair, we will have annotated comparison on each attribute

# TBD: Attribute should be involved here -- decomposition & pin-point is better
def process_dict(anno_dict):
    res_dict = {}
    for (i,j,attribute), item in anno_dict.items():
        if i not in res_dict:
            res_dict[i] = {}
        if j not in res_dict[j]:
            res_dict[i][j] = item
    return res_dict

def reverse_dict(res_dict):
    tuple_dict = {}
    for i, item in res_dict.items():
        for j, value in item.items():
            tuple_dict[(i,j)] = value
    return tuple_dict

def parse_name(name: str) -> str:
    # Replace space with _
    return name.replace(" ", "_")
def parse_date(date: str) -> str:
    # get rid of /
    return date.replace("/", "")

# Pairwise Objective Evaluation Dataset
# Goal: addition of new data will not jeopardize existing anno & data, indexing annotation is NO GO
# -- unique id for each conversation
# -- fast way to check if a 'new' conversation is already annotated
# -- pair-wise annotation dataframe will be stored with id pairs
# -- annotation will be stored in a dictionary

from src import AttributeTree, POEDataset, parse_new_conversations, parse_conversation_into_name_and_messages

# Function to handle annotation and move to next conversation
def annotate(choice):
    choice_map = {'A': [1, 0], 'B': [0, 1]}
    # Record the annotation
    st.session_state.poe_dataset.annotate(st.session_state.current_index, choice_map[choice])
    # Move to next conversation
    st.session_state.current_index += 1

# Display the conversations based on the current index
# st.set_page_config(layout="wide")
# with st.expander("See more"):
    # st.write("Additional content")
css='''
<style>
    section.main > div {max-width:50rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# st.sidebar.write("This is some content in the sidebar.")
import time

def check_anno_info_filled():
    if st.session_state.annotator_info['annotator_name'] == '' or st.session_state.annotator_info['annotation_date'] == '':
        return False
    return True

# Initialize session state variables if they don't exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = -1
if 'proceed_to_annotation' not in st.session_state:
    st.session_state.proceed_to_annotation = False
if 'annotator_info' not in st.session_state:
    st.session_state.annotator_info = {'annotator_name': '', 'annotation_date': ''}

p1 = st.empty()
p2 = st.empty()
p3 = st.empty()
p4 = st.empty()
p5 = st.empty()


annotator_name: str = 'NA'
annotation_date: str = 'NA'

if st.session_state.current_index == -1 and not st.session_state.proceed_to_annotation:

    p1.title('🙇 Welcome to TemusAI Annotation ⚓')
    # p2.markdown('<style>h2{font-size: 40px;}</style> <h2> Please tell us your name and the date</h2>', unsafe_allow_html=True)
    annotator_name = p3.text_input('Name')
    annotation_date = p4.text_input('Date')
    
    # Proceed button
    if p5.button('Proceed to Annotation'):
        st.session_state.annotator_info = {
            'annotator_name': parse_name(annotator_name), 
            'annotation_date': parse_date(annotation_date)
        }
        st.session_state.current_index = 0  # Move to the first conversation
        st.session_state.proceed_to_annotation = True

        p3.empty()
        p4.empty()
        p5.empty()
        p1.empty()

        if 'poe_dataset' not in st.session_state:
            attribute_tree = AttributeTree.make()
            annotator_info_str = f'{st.session_state.annotator_info["annotator_name"]}_{st.session_state.annotator_info["annotation_date"]}'
            st.session_state.poe_dataset = POEDataset.make(parse_new_conversations, attribute_tree, annotator_info_str)

if ('poe_dataset' in st.session_state) and (0 <= st.session_state.current_index < len(st.session_state.poe_dataset)):
    percent_complete = int(100 * (st.session_state.current_index / len(st.session_state.poe_dataset)))
    p5 = st.progress(percent_complete, text=f'{st.session_state.current_index}/{len(st.session_state.poe_dataset)}')

    dialogueA, dialogueB, attribute = st.session_state.poe_dataset[st.session_state.current_index]
    p1.title('Which customer is better in {attribute}?'.format(attribute=attribute))
    col1, col2 = p2.columns(2)

    col1.subheader('     Customer A')
    for d in dialogueA[:6]:
        name, message = parse_conversation_into_name_and_messages(d)
        col1.markdown(f'{name}: {message}')

    col2.subheader('     Customer B')
    for d in dialogueB[:6]:
        name, message = parse_conversation_into_name_and_messages(d)
        col2.markdown(f'{name}: {message}')


    # Annotation buttons
    col1.button("👍 A", on_click=lambda: annotate('A'))
    col2.button("👍 B", on_click=lambda: annotate('B'))
    
elif ('poe_dataset' in st.session_state) and (st.session_state.current_index == len(st.session_state.poe_dataset)):
    p1.markdown('<style>h1{font-size: 80px;}</style>  <h1>🙆 Great news!</h1>', unsafe_allow_html=True)
    # placeholder.markdown('<h1>🙆 Great news!</h1>', unsafe_allow_html=True)
    p2.markdown('<style>h2{font-size: 40px;}</style> <h2> All conversations annotated</h2>', unsafe_allow_html=True)
    # placeholder.markdown('<h2> All conversations annotated</h2>', unsafe_allow_html=True)

    st.session_state.poe_dataset.save()
