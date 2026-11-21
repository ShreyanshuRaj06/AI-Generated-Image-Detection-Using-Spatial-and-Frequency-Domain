import re
import numpy as np
from collections import Counter

# Set of common English function words
FUNCTION_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t',
    'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is',
    'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
    'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
    'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they',
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
    'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

def analyze_prose(tex_text):
    # Strip latex commands, environments, math
    clean = re.sub(r'%.*$', '', tex_text, flags=re.MULTILINE)
    clean = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\\begin\{align\}.*?\\end\{align\}', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\$[^$]*\$', '', clean)
    clean = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', clean)
    clean = re.sub(r'[{}]', '', clean)

    # Sentences
    raw_sentences = re.split(r'[.!?]+', clean)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
    
    words_per_sent = [len(s.split()) for s in sentences]
    all_words = [w.lower().strip(".,;:\"'()[]") for s in sentences for w in s.split()]
    all_words = [w for w in all_words if w and not w.isdigit()]

    # 1. Mean sentence length
    mean_len = np.mean(words_per_sent) if words_per_sent else 0
    
    # 2. Percentage of 3-8 word sentences
    short_sents = [l for l in words_per_sent if 3 <= l <= 8]
    pct_short = (len(short_sents) / len(words_per_sent)) * 100 if words_per_sent else 0

    # 3. Percentage of 41+ word sentences
    long_sents = [l for l in words_per_sent if l >= 41]
    pct_long = (len(long_sents) / len(words_per_sent)) * 100 if words_per_sent else 0

    # 4. Function word ratio
    func_count = sum(1 for w in all_words if w in FUNCTION_WORDS)
    func_ratio = func_count / len(all_words) if all_words else 0

    # 5. Mean word length
    char_lengths = [len(w) for w in all_words]
    mean_word_len = np.mean(char_lengths) if char_lengths else 0

    # 6. 3-gram repetition rate
    trigrams = [tuple(all_words[i:i+3]) for i in range(len(all_words)-2)]
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(c for c in trigram_counts.values() if c > 1)
    trigram_rep_rate = (repeated_trigrams / len(trigrams)) * 100 if trigrams else 0

    # 7. Stylized starters
    banned_starters = ["crucially", "fundamentally", "notably", "interestingly", "importantly", "specifically", "consequently"]
    starters_found = []
    for s in sentences:
        first_word = s.split()[0].lower().strip(".,;:\"'")
        if first_word in banned_starters:
            starters_found.append(first_word)

    print("=" * 60)
    print("DETECTOR METRIC REPORT (CURRENT MANUSCRIPT)")
    print("=" * 60)
    print(f"Total sentences:             {len(sentences)}")
    print(f"Mean sentence length:        {mean_len:.1f} words  (Target: 18.0 - 23.0; Human avg: 23.2, AI: 29.2)")
    print(f"Short sentences (3-8 words): {pct_short:.1f}%       (Target: > 6.0%; Human: 5.8%, AI: 2.4%)")
    print(f"Long sentences (41+ words):  {pct_long:.1f}%       (Target: < 5.0%; Human: 6.2%, AI: 17.0%)")
    print(f"Function word ratio:         {func_ratio:.3f}      (Target: > 0.395; Human: 0.402, AI: 0.330)")
    print(f"Mean word length:            {mean_word_len:.2f} chars (Target: < 5.35; Human: 5.24, AI: 5.86)")
    print(f"3-gram repetition rate:      {trigram_rep_rate:.2f}%     (Target: > 3.0%; Human: 3.5%, AI: 1.4%)")
    print(f"Stylized starters found:     {starters_found or 'NONE ✅'}")
    print("=" * 60)

with open("paper/main.tex", "r", encoding="utf-8") as f:
    analyze_prose(f.read())
