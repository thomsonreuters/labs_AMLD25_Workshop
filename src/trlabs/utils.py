import re
import html
import difflib
from IPython.display import HTML

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Decode HTML entities (like < > &amp; etc)
    text = html.unescape(text)

    # Remove multiple spaces and newlines
    text = ' '.join(text.split())

    # Remove 'Reuter' variations at the end
    text = text.replace(' Reuter', '').replace(' Reuters', '')

    # Remove any remaining multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_tag_content(text: str, tag_name: str) -> str:
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""

def create_html_diff_with_background(a: str, b: str) -> HTML:
    """
    Create HTML diff between two strings with background highlighting
    """
    # Create a diff object
    d = difflib.Differ()
    diff = list(d.compare(a.split(), b.split()))
    
    # Create HTML with colored backgrounds
    html = []
    for word in diff:
        if word.startswith('  '):  # unchanged
            html.append(word[2:])
        elif word.startswith('- '):  # deletion
            html.append(f'<span style="background-color: #ffb3ba">{word[2:]}</span>')
        elif word.startswith('+ '):  # addition
            html.append(f'<span style="background-color: #88c8f7">{word[2:]}</span>')
    
    return HTML(' '.join(html))

def dataset_creation(row, system_prompt, instruction):
    row["cleaned_text"] = clean_text(row["text"])
    row["system"] = [{"role":"system", "content": system_prompt}]
    row["prompt"] = row["cleaned_text"] + "\n\n" + instruction
    row["messages"] = [{'content': row["prompt"], 'role': 'user'}]
    return row

def reuters_cleaning_dataset(example):
    diff = example["chosen_reward"] - example["rejected_reward"]
    if example["chosen_reward"]>=3 and  diff>1:
        return True
    else:
        return False

def ultrafeedback_cleaning_dataset(example):
    diff = example["score_chosen"] - example["score_rejected"]
    if example["score_chosen"]>=8 and  diff>2:
        return True
    else:
        return False

def not_relevant_data(row):
    if row["title"] == "PROPOSED OFFERINGS RECENTLY FILED WITH THE SEC" or \
       row["title"] == "FED ADDS RESERVES VIA CUSTOMER REPURCHASE":
       return False
    return True 