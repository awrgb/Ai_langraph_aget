import os
import re
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

# Load environment variables
load_dotenv()

# Configure the generative AI model using the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Setup the LLM for language generation tasks
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-1.5-flash")

# State typing for essay grading
class State(TypedDict):
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float
    suggested_edits: str
    topic: str

# Function to extract score from LLM response
def extract_score(content: str) -> float:
    """Extract the numeric score from the LLM's response."""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    return float(match.group(1)) if match else 0.0

# Grading workflow
def grade_essay(essay: str, topic: str) -> dict:
    """Grade the given essay."""
    state = State(
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0,
        suggested_edits="",
        topic=topic,
    )

    # Sequentially check relevance, grammar, structure, and depth
    state = check_relevance(state)
    if state['relevance_score'] > 1:
        state = check_grammar(state)
    if state['grammar_score'] > 1:
        state = analyze_structure(state)
    if state['structure_score'] > 1:
        state = evaluate_depth(state)

    # Calculate final score
    state = calculate_final_score(state)

    return state

# Grading functions
def check_relevance(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevancy of the essay to the topic provided by the user: {topic}. "
        "Provide a relevance score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"], topic=state["topic"]))
    state["relevance_score"] = extract_score(result)
    return state

def check_grammar(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the grammar of the essay. Provide a grammar score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["grammar_score"] = extract_score(result)
    return state

def analyze_structure(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the structure of the essay. Provide a structure score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["structure_score"] = extract_score(result)
    return state

def evaluate_depth(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the depth of analysis in the essay. Provide a depth score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["depth_score"] = extract_score(result)
    return state

def suggest_edits(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Suggest improvements to the essay based on grammar, relevance, and structure.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["suggested_edits"] = result
    return state

def calculate_final_score(state: State) -> State:
    state = suggest_edits(state)
    state["final_score"] = (
        state["relevance_score"] +
        state["grammar_score"] +
        state["structure_score"] +
        state["depth_score"]
    )
    return state

def generate_file(result: dict, additional_comments: str) -> None:
    """Generate a text file with the grading results and additional comments."""
    with open("grading_results.txt", "w") as file:
        file.write(f"Final Essay Score: {result['final_score']:.1f}/10\n")
        file.write(f"Relevance Score: {result['relevance_score']:.1f}\n")
        file.write(f"Grammar Score: {result['grammar_score']:.1f}\n")
        file.write(f"Structure Score: {result['structure_score']:.1f}\n")
        file.write(f"Depth Score: {result['depth_score']:.1f}\n")
        file.write(f"Suggested Improvements: {result['suggested_edits']}\n")
        file.write(f"Additional Comments: {additional_comments}\n")

# Streamlit UI
st.title("🌟 Essay Grading Application 🌟")

# Input for topic
st.text_input("Enter the topic:", key="topic")

# Input for essay
sample_essay = st.text_area("Please enter your essay:")

# Submit button
if st.button("Grade Essay"):
    if st.session_state.topic:
        if sample_essay:
            result = grade_essay(sample_essay, st.session_state.topic)

            # Display results
            st.markdown(f"### **Final Essay Score:** {result['final_score']:.1f}/10")
            st.markdown(f"**Relevance Score:** {result['relevance_score']:.1f}")
            st.markdown(f"**Grammar Score:** {result['grammar_score']:.1f}")
            st.markdown(f"**Structure Score:** {result['structure_score']:.1f}")
            st.markdown(f"**Depth Score:** {result['depth_score']:.1f}")
            st.markdown(f"**Suggested Improvements:** {result['suggested_edits']}")
        else:
            st.error("Please enter your essay.")
    else:
        st.error("Please enter the topic.")
