import os
import re
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate

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
    tone_score: float
    citation_score: float
    word_count: int
    plagiarism_check: str
    personalized_tips: str

# Extract numeric score from LLM response
def extract_score(content: str) -> float:
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    return float(match.group(1)) if match else 0.0

# Real-time suggestions and grading workflow
def grade_essay(essay: str, topic: str) -> dict:
    state = State(
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0,
        suggested_edits="",
        topic=topic,
        tone_score=0.0,
        citation_score=0.0,
        word_count=len(essay.split()),
        plagiarism_check="",
        personalized_tips=""
    )

    # Sequentially check relevance, grammar, structure, and depth
    state = check_relevance(state)
    if state['relevance_score'] > 1:
        state = check_grammar(state)
    if state['grammar_score'] > 1:
        state = analyze_structure(state)
    if state['structure_score'] > 1:
        state = evaluate_depth(state)

    # Additional checks
    state = check_tone(state)
    state = check_citations(state)
    state = plagiarism_check(state)

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

def check_tone(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the tone and formality of the essay. Provide a tone score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["tone_score"] = extract_score(result)
    return state

def check_citations(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the citation and reference usage. Provide a citation score between 0 and 2.5.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["citation_score"] = extract_score(result)
    return state

def plagiarism_check(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Check for any potential plagiarism in the following essay. Provide a yes/no response.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    state["plagiarism_check"] = result.strip()
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
        state["depth_score"] +
        state["tone_score"] +
        state["citation_score"]
    )
    return state

# Streamlit UI
st.title("💡 Cutting-Edge Essay Grading with Real-Time Suggestions")

# Input for topic
st.text_input("Enter the topic:", key="topic")

# Input for essay
sample_essay = st.text_area("Start typing your essay:")

# Word count display
if sample_essay:
    word_count = len(sample_essay.split())
    st.write(f"**Word Count:** {word_count}")

# Submit button
if st.button("Analyze Essay"):
    if st.session_state.topic:
        if sample_essay:
            result = grade_essay(sample_essay, st.session_state.topic)

            # Display results with real-time suggestions
            st.markdown(f"### **Final Essay Score:** {result['final_score']:.1f}/15")
            st.markdown(f"**Relevance Score:** {result['relevance_score']:.1f}")
            st.markdown(f"**Grammar Score:** {result['grammar_score']:.1f}")
            st.markdown(f"**Structure Score:** {result['structure_score']:.1f}")
            st.markdown(f"**Depth Score:** {result['depth_score']:.1f}")
            st.markdown(f"**Tone Score:** {result['tone_score']:.1f}")
            st.markdown(f"**Citation Score:** {result['citation_score']:.1f}")
            st.markdown(f"**Plagiarism Check:** {result['plagiarism_check']}")
            st.markdown(f"**Suggested Improvements:** {result['suggested_edits']}")

            # Real-time suggestions and feedback
            st.info(f"**Suggestions:** {result['suggested_edits']}")
        else:
            st.error("Please enter your essay.")
    else:
        st.error("Please enter the topic.")
