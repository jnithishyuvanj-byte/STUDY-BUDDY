import streamlit as st
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import difflib

# Ensure required NLTK data is present; download only if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass


def split_sentences(text: str):
    """Return a list of sentences for the given text.

    Prefer NLTK's sent_tokenize but fall back to a simple regex splitter
    if the punkt resource is not available or raises a LookupError.
    """
    try:
        return sent_tokenize(text)
    except Exception:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]


# --- Feature functions ---

def explain_topic(topic, text=""):
    if text:
        sentences = split_sentences(text)
        relevant_sentences = [s for s in sentences if topic.lower() in s.lower()]
        if relevant_sentences:
            context = " ".join(relevant_sentences[:2])
            return f"Based on your notes: {context}\n\nIn simple terms, {topic} refers to the concept described in your notes."

    explanations = [
        f"{topic} is an important concept. It can be understood as the process or idea related to {topic.lower()} in simple terms.",
        f"To understand {topic}, imagine a simple example that shows how it works in real life.",
        f"In short, {topic} means the basic idea or principle behind {topic.lower()} that helps in many applications.",
        f"The topic '{topic}' can be explained easily — it represents something that helps us learn and grow knowledge about {topic.lower()}."
    ]
    return random.choice(explanations)


def summarize_text(text, summary_ratio=0.2):
    """Extractive summarizer based on word frequency.

    summary_ratio: fraction of sentences to include (0 < ratio <= 1)
    """
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return text

    words = re.findall(r"\b\w+\b", text.lower())
    stop_words = set()
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # If stopwords not available, use a small fallback set
        stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a'}

    freq = {}
    for w in words:
        if w in stop_words or len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1

    # score sentences
    sent_scores = []
    for s in sentences:
        s_words = re.findall(r"\b\w+\b", s.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        sent_scores.append((s, score))

    # choose top N sentences by score
    n = max(3, int(len(sentences) * summary_ratio))
    top_sentences = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:n]

    # preserve original order
    top_set = set(s for s, _ in top_sentences)
    ordered = [s for s in sentences if s in top_set]
    return " ".join(ordered)


def generate_quiz(text, num_questions=3):
    """Return a list of quiz items as dicts: {'q','a','keys'}."""
    if not text:
        return []

    sentences = split_sentences(text)
    words = re.findall(r"\b\w+\b", text.lower())
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a'}

    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1

    top_words = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]]

    quiz = []
    used = set()
    for i in range(num_questions):
        if top_words:
            topic = random.choice(top_words)
            relevant = [s for s in sentences if topic in s.lower() and s not in used]
            if relevant:
                answer = relevant[0].strip()
                q = f"What is mentioned about '{topic}' in the notes?"
                used.add(answer)
            else:
                # fallback to random unused sentence
                candidates = [s for s in sentences if s not in used]
                if not candidates:
                    candidates = sentences
                answer = random.choice(candidates).strip()
                q = f"Explain the meaning of: '{answer[:60]}...'"
                used.add(answer)
        else:
            answer = random.choice(sentences).strip()
            q = f"Explain the meaning of: '{answer[:60]}...'"

        ans_words = re.findall(r"\b\w+\b", answer.lower())
        keys = []
        for w in ans_words:
            if w not in stop_words and len(w) > 3 and w not in keys:
                keys.append(w)
            if len(keys) >= 3:
                break

        quiz.append({'q': q, 'a': answer, 'keys': keys})

    return quiz


def grade_answer(user_ans, correct_ans, keys=None):
    """Return (is_correct: bool, score: float) using keyword checking and similarity."""
    user = user_ans.lower() if user_ans else ""
    correct = correct_ans.lower() if correct_ans else ""

    # keyword match
    key_hits = 0
    if keys:
        for k in keys:
            if k in user:
                key_hits += 1
    key_score = key_hits / max(1, len(keys or []))

    # similarity
    sim = difflib.SequenceMatcher(None, user, correct).ratio()

    # weighted decision
    score = 0.6 * key_score + 0.4 * sim
    return (score >= 0.5, score)


def main():
    st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")

    st.title("📚 AI Study Buddy — Offline")
    st.write("Upload a .txt file with your notes, then use the sidebar tools to summarize, explain, or generate quizzes.")

    if 'notes_text' not in st.session_state:
        st.session_state['notes_text'] = ""

    uploaded = st.file_uploader("Upload notes (.txt)", type=['txt'], key='notes_upload')
    if uploaded is not None:
        text = uploaded.read().decode('utf-8')
        st.session_state['notes_text'] = text
        st.success(f"Loaded notes — {len(text)} characters")

    # Show notes
    if st.session_state['notes_text']:
        with st.expander("View notes (read-only)"):
            st.text_area("Notes content", value=st.session_state['notes_text'], height=300, key='notes_viewer', disabled=True, label_visibility='collapsed')

    st.sidebar.header("Tools")

    # Summarize options
    st.sidebar.subheader("Summarize")
    percent = st.sidebar.slider("Summary length (%)", min_value=10, max_value=60, value=20, step=5, key='summary_ratio')
    ratio = percent / 100  # Convert percentage to ratio
    if st.sidebar.button("Summarize Notes", key='summarize_btn'):
        if st.session_state['notes_text']:
            summary = summarize_text(st.session_state['notes_text'], summary_ratio=ratio)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.sidebar.warning("Upload notes first.")

    # Explain
    st.sidebar.subheader("Explain Topic")
    topic = st.sidebar.text_input("Topic to explain", key='topic_input')
    if st.sidebar.button("Explain", key='explain_btn') and topic:
        explanation = explain_topic(topic, st.session_state.get('notes_text', ''))
        st.subheader(f"Explanation — {topic}")
        st.write(explanation)

    # Quiz generation
    st.sidebar.subheader("Quiz")
    q_count = st.sidebar.number_input("Number of questions", min_value=1, max_value=10, value=3, key='quiz_count')
    if st.sidebar.button("Generate Quiz", key='quiz_btn'):
        if st.session_state['notes_text']:
            st.session_state['quiz'] = generate_quiz(st.session_state['notes_text'], num_questions=q_count)
            st.session_state['quiz_submitted'] = False
        else:
            st.sidebar.warning("Upload notes first.")

    # Render quiz if present
    if st.session_state.get('quiz'):
        st.subheader("Quiz")
        quiz = st.session_state['quiz']
        if not isinstance(quiz, list):
            st.write(quiz)
        else:
            answers = []
            with st.form("quiz_form"):
                for i, item in enumerate(quiz):
                    st.write(f"Q{i+1}. {item['q']}")
                    ans = st.text_area(f"Your answer for Q{i+1}", key=f"ans_{i}", height=80)
                    answers.append(ans)
                submitted = st.form_submit_button("Submit Answers")
                if submitted:
                    st.session_state['quiz_submitted'] = True
                    score_total = 0.0
                    for i, item in enumerate(quiz):
                        correct = item.get('a', '')
                        keys = item.get('keys', [])
                        is_correct, sc = grade_answer(st.session_state.get(f"ans_{i}"), correct, keys)
                        score_total += sc
                        score_percent = int(sc * 100)
                        st.write(f"Q{i+1} — Score: {score_percent}% — {'Correct' if is_correct else 'Partial/Incorrect'}")
                        if not is_correct:
                            st.write(f"Correct answer: {correct}")
                    avg_percent = int((score_total / len(quiz) if quiz else 0) * 100)
                    st.success(f"Quiz average score: {avg_percent}%")

    st.markdown("---")


if __name__ == '__main__':
    main()
